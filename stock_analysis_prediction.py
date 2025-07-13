# 安装必要库
# pip install dash pandas yahoofinancials plotly scikit-learn ta
import os
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from openpyxl.styles.colors import BLACK
from qtconsole.mainwindow import background
from yahoofinancials import YahooFinancials
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ta import add_all_ta_features
import time
import json
import hashlib
from pathlib import Path
import warnings
from io import StringIO

# 忽略Pandas的setitem警告
warnings.filterwarnings('ignore', category=FutureWarning, message='.*Series.__setitem__.*')

# 初始化Dash应用
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# 默认股票列表
default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'JPM', 'V']

# 创建缓存目录
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

# 应用布局
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("股票分析与预测系统", className="text-center mt-3 mb-4 text-primary"),
            html.P("使用机器学习模型预测股票走势", className="text-center text-light")
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("股票选择与参数设置", className="bg-primary text-white"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("选择股票代码", className="mb-1"),
                            dcc.Dropdown(
                                id='symbol-selector',
                                options=[{'label': s, 'value': s} for s in default_symbols],
                                value='AAPL',
                                clearable=False,
                                style={'color': 'black'}
                            )
                        ], width=6),

                        dbc.Col([
                            html.Label("数据时间范围", className="mb-1"),
                            dcc.Dropdown(
                                id='period-selector',
                                options=[
                                    {'label': '1个月', 'value': '1mo'},
                                    {'label': '3个月', 'value': '3mo'},
                                    {'label': '6个月', 'value': '6mo'},
                                    {'label': '1年', 'value': '1y'},
                                    {'label': '3年', 'value': '3y'},
                                    {'label': '5年', 'value': '5y'},
                                    {'label': '最大范围', 'value': 'max'}
                                ],
                                value='1y',
                                clearable=False
                            )
                        ], width=6)
                    ]),

                    dbc.Row([
                        dbc.Col([
                            html.Label("预测天数", className="mb-1 mt-2"),
                            dcc.Slider(
                                id='forecast-days',
                                min=1,
                                max=30,
                                step=1,
                                value=7,
                                marks={i: str(i) for i in range(0, 31, 5)}
                            )
                        ], width=12)
                    ]),

                    dbc.Row([
                        dbc.Col([
                            dbc.Button("获取数据", id='fetch-data', color="primary", className="mt-3 w-100")
                        ], width=12)
                    ]),

                    dbc.Row([
                        dbc.Col([
                            html.Div(id='stock-info', className="mt-4 p-3 bg-dark rounded")
                        ], width=12)
                    ])
                ])
            ], className="mb-4")
        ], width=12, lg=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("股票价格图表", className="bg-primary text-white"),
                dbc.CardBody([
                    dcc.Graph(id='price-chart', className="mb-3"),
                    dcc.Loading(id="loading-chart", type="circle")
                ])
            ], className="mb-4")
        ], width=12, lg=8)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("技术指标分析", className="bg-primary text-white"),
                dbc.CardBody([
                    dcc.Graph(id='indicator-chart'),
                    dcc.Loading(id="loading-indicators", type="circle")
                ])
            ])
        ], width=12, lg=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("价格预测", className="bg-primary text-white"),
                dbc.CardBody([
                    dcc.Graph(id='forecast-chart'),
                    dcc.Loading(id="loading-forecast", type="circle")
                ])
            ])
        ], width=12, lg=6)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("股票基本面数据", className="bg-primary text-white"),
                dbc.CardBody([
                    html.Div(id='fundamentals-table', className="mt-2")
                ])
            ])
        ], width=12)
    ], className="mt-4"),

    dcc.Store(id='stock-data'),
    dcc.Store(id='stock-info-store')
], fluid=True)


def get_cache_key(symbol, period):
    """生成唯一的缓存键"""
    return hashlib.md5(f"{symbol}_{period}".encode()).hexdigest()


def save_to_cache(key, data):
    """保存数据到缓存"""
    cache_file = CACHE_DIR / f"{key}.json"
    with open(cache_file, 'w') as f:
        json.dump({
            'df': data[0].to_json(orient='split'),
            'info': data[1].to_dict()
        }, f)


def load_from_cache(key):
    """从缓存加载数据"""
    cache_file = CACHE_DIR / f"{key}.json"
    if not cache_file.exists():
        return None

    with open(cache_file, 'r') as f:
        data = json.load(f)
        # 使用StringIO包装JSON字符串
        df = pd.read_json(StringIO(data['df']), orient='split')
        info_df = pd.DataFrame(data['info'])
        return df, info_df


def period_to_date_range(period):
    """将时间范围字符串转换为日期范围"""
    end_date = datetime.now()

    if period == '1mo':
        start_date = end_date - timedelta(days=30)
    elif period == '3mo':
        start_date = end_date - timedelta(days=90)
    elif period == '6mo':
        start_date = end_date - timedelta(days=180)
    elif period == '1y':
        start_date = end_date - timedelta(days=365)
    elif period == '3y':
        start_date = end_date - timedelta(days=3 * 365)
    elif period == '5y':
        start_date = end_date - timedelta(days=5 * 365)
    else:  # max
        start_date = end_date - timedelta(days=10 * 365)  # 10年作为最大范围

    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


# 获取股票数据
def fetch_stock_data(symbol, period):
    """获取股票历史数据和基本信息，带缓存和重试机制"""
    # 检查缓存
    cache_key = get_cache_key(symbol, period)
    cached_data = load_from_cache(cache_key)
    if cached_data:
        print(f"Use cached data: {symbol} {period}")
        return cached_data

    # 设置重试机制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 使用yahoofinancials获取数据
            stock = YahooFinancials(symbol)

            # 获取日期范围
            start_date, end_date = period_to_date_range(period)

            # 获取历史价格数据
            price_data = stock.get_historical_price_data(start_date, end_date, "daily")
            print(f"price_data:{price_data}")
            # 检查数据是否存在
            if symbol not in price_data or not price_data[symbol]['prices']:
                print(f"无价格数据: {symbol} {period}")
                return None, None

            # 转换数据为DataFrame
            df = pd.DataFrame(price_data[symbol]['prices'])

            # 处理日期并设为索引
            df['formatted_date'] = pd.to_datetime(df['formatted_date'])
            df = df.set_index('formatted_date')
            df.index.name = 'Date'

            # 重命名列以符合ta库要求
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # 确保所有必需的列都存在
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    print(f"缺少列: {col}")
                    return None, None

            # 添加技术指标
            df = add_all_ta_features(
                df,
                open="Open",
                high="High",
                low="Low",
                close="Close",
                volume="Volume"
            )

            # 获取基本信息
            summary_data = stock.get_summary_data()
            #获取股票报价数据
            stock_quote_type_data = stock.get_stock_quote_type_data()
            #获取股票关键数据统计
            key_statistics_data = stock.get_key_statistics_data()
            # 处理基本信息
            if symbol in summary_data:
                stock_info = summary_data[symbol]
                stock_quote_type_data_info = stock_quote_type_data[symbol]
                key_statistics_data_info = key_statistics_data[symbol]
                info_df = pd.DataFrame({
                    '指标': ['公司名称', '行业', '市值', '市盈率', '市净率', '股息率',
                             '52周最高', '52周最低', 'Beta值', '平均成交量'],
                    '值': [
                        stock_quote_type_data_info.get('longName', 'N/A'),
                        stock_info.get('sector', 'N/A'),
                        f"${stock_info.get('marketCap', 'N/A'):,.0f}" if stock_info.get('marketCap') else 'N/A',
                        stock_info.get('trailingPE', 'N/A'),
                        key_statistics_data_info.get('priceToBook', 'N/A'),
                        f"{stock_info.get('dividendRate', 0) * 100:.2f}%" if stock_info.get('dividendRate') else '0%',
                        f"${stock_info.get('fiftyTwoWeekHigh', 'N/A'):,.2f}",
                        f"${stock_info.get('fiftyTwoWeekLow', 'N/A'):,.2f}",
                        stock_info.get('beta', 'N/A'),
                        f"{stock_info.get('averageVolume', 'N/A'):,.0f}" if stock_info.get('averageVolume') else 'N/A'
                    ]
                })
            else:
                print(f"无摘要数据: {symbol}")
                info_df = pd.DataFrame({
                    '指标': ['错误'],
                    '值': ['无法获取基本信息']
                })

            # 保存到缓存
            save_to_cache(cache_key, (df, info_df))

            return df, info_df

        except Exception as e:
            print(f"try {attempt + 1}/{max_retries} fail: {e}")
            # 如果是速率限制错误，等待一段时间再重试
            if "Too Many Requests" in str(e) or "Rate limited" in str(e):
                wait_time = (attempt + 1) * 5  # 指数退避
                print(f"速率限制触发，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            else:
                # 其他错误直接返回
                return None, None

    print(f"获取 {symbol} 数据失败，已达最大重试次数")
    return None, None


# 训练预测模型 - 修复索引警告
def train_prediction_model(df, forecast_days):
    """使用随机森林预测未来价格"""
    try:
        # 准备数据 - 使用copy()避免链式赋值警告
        data = df.copy()
        data.to_csv(f"C:\\D\\stock\\data\\{forecast_days}.csv", index=False)

        key_columns = ['Close', 'Volume', 'volatility_bbh', 'volatility_bbl',
            'volatility_bbm', 'volatility_bbw', 'volatility_kcc',
            'volatility_kch', 'volatility_kcl', 'trend_macd',
            'trend_macd_signal', 'momentum_rsi', 'volume_adi',
            'volume_obv']
        #过滤数据为NaN的数据,数据为NaN的删除行
        data = data.dropna(axis=0,subset=key_columns)
        data.to_csv(f"C:\\D\\stock\\data\\data2025.csv",index=False)
        # 检查数据量是否足够
        min_samples = max(20, forecast_days * 2)  # 至少20个样本或预测天数的2倍

        if len(data) < min_samples:
            print(f"数据不足: {len(data)} < {min_samples}")
            return None, None

        # 创建预测目标 - 使用iloc避免位置索引警告
        # 这里我们使用iloc来按位置创建目标列
        close_values = data['Close'].values

        target = np.empty(len(close_values))

        # target[:] = np.nan

        target[:-forecast_days] = close_values[forecast_days:]

        # 将目标列添加到DataFrame
        data = data.assign(target=target)
        data = data.dropna(axis=0,subset=key_columns)

        # 再次检查数据量是否足够
        if len(data) < 10:  # 至少需要10个样本
            print(f"处理后数据不足: {len(data)}")
            return None, None

        # 选择特征
        features = [
            'Close', 'Volume', 'volatility_bbh', 'volatility_bbl',
            'volatility_bbm', 'volatility_bbw', 'volatility_kcc',
            'volatility_kch', 'volatility_kcl', 'trend_macd',
            'trend_macd_signal', 'momentum_rsi', 'volume_adi',
            'volume_obv'
        ]

        # 确保所有特征都存在
        available_features = [f for f in features if f in data.columns]
        if not available_features:
            available_features = ['Close', 'Volume']  # 回退到基本特征

        x = data[available_features]

        y = data['target']

        # 划分训练测试集
        # 确保有足够的样本进行划分
        if len(x) < 2:
            print(f"样本数量不足: {len(x)}")
            return None, None

        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, shuffle=False
        )

        # 训练模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 评估模型
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # 预测未来 - 使用iloc获取最后一行
        last_data = data.iloc[[-1]][available_features].values.reshape(1, -1)
        future_price = model.predict(last_data)[0]

        # 创建预测数据框
        last_date = data.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted': [future_price] * forecast_days
        })

        return future_df, rmse

    except Exception as e:
        print(f"预测错误: {e}")
        return None, None


# 回调函数
@app.callback(
    [Output('stock-data', 'data'),
     Output('stock-info-store', 'data'),
     Output('stock-info', 'children')],
    [Input('fetch-data', 'n_clicks')],
    [State('symbol-selector', 'value'),
     State('period-selector', 'value')]
)
def update_stock_data(n_clicks, symbol, period):
    """获取并存储股票数据"""
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update

    df, info_df = fetch_stock_data(symbol, period)

    if df is None or info_df is None:
        return dash.no_update, dash.no_update, html.Div("无法获取数据，请稍后再试", className="text-danger")

    # 创建基本信息显示
    info_card = dbc.Table.from_dataframe(
        info_df,
        striped=True,
        bordered=True,
        hover=True,
        className="text-light",
        style={'fontSize': '0.85rem'}
    )

    return df.to_json(date_format='iso', orient='split'), info_df.to_json(orient='split'), info_card


@app.callback(
    Output('price-chart', 'figure'),
    [Input('stock-data', 'data')],
    [State('symbol-selector', 'value')]
)
def update_price_chart(stock_data, symbol):
    """更新价格图表"""
    if stock_data is None:
        return go.Figure()

    try:
        # 使用StringIO包装JSON字符串
        df = pd.read_json(StringIO(stock_data), orient='split')

        df.index = pd.to_datetime(df.index)

        # 创建K线图
        fig = go.Figure(data=[
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='价格'
            )
        ])

        # 添加移动平均线
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()

        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA20'],
            line=dict(color='orange', width=1.5),
            name='20日均线'
        ))

        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA50'],
            line=dict(color='blue', width=1.5),
            name='50日均线'
        ))

        # 更新布局
        fig.update_layout(
            title=f'{symbol} 股票价格走势',
            xaxis_title='日期',
            yaxis_title='价格 (USD)',
            template='plotly_dark',
            height=450,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig
    except Exception as e:
        print(f"更新价格图表错误: {e}")
        return go.Figure()


@app.callback(
    Output('indicator-chart', 'figure'),
    [Input('stock-data', 'data')],
    [State('symbol-selector', 'value')]
)
def update_indicator_chart(stock_data, symbol):
    """更新技术指标图表"""
    if stock_data is None:
        return go.Figure()

    try:
        # 使用StringIO包装JSON字符串
        df = pd.read_json(StringIO(stock_data), orient='split')
        df.index = pd.to_datetime(df.index)

        # 创建子图
        fig = go.Figure()

        # 添加布林带（如果存在）
        if 'volatility_bbh' in df.columns and 'volatility_bbl' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['volatility_bbh'],
                line=dict(color='rgba(255, 255, 255, 0.5)', width=1),
                name='布林带上轨'
            ))

            fig.add_trace(go.Scatter(
                x=df.index, y=df['volatility_bbl'],
                line=dict(color='rgba(255, 255, 255, 0.5)', width=1),
                name='布林带下轨',
                fill='tonexty',
                fillcolor='rgba(50, 50, 50, 0.2)'
            ))

        # 添加收盘价
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            line=dict(color='#1f77b4', width=2),
            name='收盘价'
        ))

        # 添加RSI（如果存在）
        if 'momentum_rsi' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['momentum_rsi'],
                line=dict(color='green', width=2),
                name='RSI',
                yaxis='y2'
            ))

        # 添加MACD（如果存在）
        if 'trend_macd' in df.columns and 'trend_macd_signal' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['trend_macd'],
                line=dict(color='yellow', width=1.5),
                name='MACD',
                yaxis='y3'
            ))

            fig.add_trace(go.Scatter(
                x=df.index, y=df['trend_macd_signal'],
                line=dict(color='red', width=1.5),
                name='信号线',
                yaxis='y3'
            ))

        # 更新布局
        layout_updates = {
            'title': f'{symbol} 技术指标分析',
            'template': 'plotly_dark',
            'height': 450,
            'legend': dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            'yaxis': dict(title='价格', domain=[0.55, 1.0]),
            'xaxis': dict(domain=[0, 0.95])
        }

        if 'momentum_rsi' in df.columns:
            layout_updates['yaxis2'] = dict(title='RSI', domain=[0.35, 0.5], range=[0, 100])

        if 'trend_macd' in df.columns:
            layout_updates['yaxis3'] = dict(title='MACD', domain=[0.0, 0.3])

        fig.update_layout(**layout_updates)

        # 添加RSI参考线
        if 'momentum_rsi' in df.columns:
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, yref="y2")
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, yref="y2")

        return fig
    except Exception as e:
        print(f"更新技术指标图表错误: {e}")
        return go.Figure()


@app.callback(
    [Output('forecast-chart', 'figure'),
     Output('fundamentals-table', 'children')],
    [Input('stock-data', 'data'),
     Input('forecast-days', 'value'),
     Input('stock-info-store', 'data')],
    [State('symbol-selector', 'value')]
)
def update_forecast_chart(stock_data, forecast_days, info_data, symbol):

    """更新预测图表和基本面数据"""
    if stock_data is None or info_data is None:
        return go.Figure(), dash.no_update

    try:
        # 使用StringIO包装JSON字符串
        df = pd.read_json(StringIO(stock_data), orient='split')
        df.index = pd.to_datetime(df.index)

        # 使用StringIO包装JSON字符串
        info_df = pd.read_json(StringIO(info_data), orient='split')

        # 训练模型并获取预测
        future_df, rmse = train_prediction_model(df, forecast_days)

        # 创建图表
        fig = go.Figure()

        # 添加历史数据
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            line=dict(color='#1f77b4', width=2),
            name='历史价格'
        ))

        # 如果有预测数据
        if future_df is not None:
            # 添加预测数据
            fig.add_trace(go.Scatter(
                x=future_df['Date'], y=future_df['Predicted'],
                line=dict(color='green', width=2, dash='dash'),
                name=f'{forecast_days}天预测'
            ))

            # 添加预测点
            fig.add_trace(go.Scatter(
                x=[future_df['Date'].iloc[-1]],
                y=[future_df['Predicted'].iloc[-1]],
                mode='markers',
                marker=dict(color='red', size=10),
                name='预测价格'
            ))

            # 添加误差区间
            if rmse:
                fig.add_trace(go.Scatter(
                    x=future_df['Date'],
                    y=future_df['Predicted'] + rmse,
                    line=dict(color='rgba(255, 255, 255, 0.1)'),
                    showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=future_df['Date'],
                    y=future_df['Predicted'] - rmse,
                    line=dict(color='rgba(255, 255, 255, 0.1)'),
                    fill='tonexty',
                    fillcolor='rgba(50, 50, 50, 0.2)',
                    name='置信区间'
                ))

            title = f'{symbol} 未来{forecast_days}天价格预测'
        else:
            # 如果没有预测数据，显示错误信息
            title = f'{symbol} 预测失败（数据不足）'
            fig.add_annotation(
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                text="预测失败：数据不足",
                showarrow=False,
                font=dict(size=20, color="red")
            )

        # 更新布局
        fig.update_layout(
            title=title,
            xaxis_title='日期',
            yaxis_title='价格 (USD)',
            template='plotly_dark',
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # 创建基本面数据表
        fundamentals_table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in info_df.columns],
            data=info_df.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': '#2c3e50',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_cell={
                'backgroundColor': '#1a1a1a',
                'color': 'white',
                'textAlign': 'left',
                'minWidth': '100px'
            },
            style_cell_conditional=[
                {'if': {'column_id': '指标'},
                 'fontWeight': 'bold',
                 'width': '30%'}
            ]
        )

        return fig, fundamentals_table
    except Exception as e:
        print(f"更新预测图表错误: {e}")
        return go.Figure(), html.Div("处理数据时发生错误", className="text-danger")


# 运行应用
if __name__ == '__main__':
    app.run(debug=True, port=8030)
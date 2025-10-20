"""
AI Data Science Analyzer
Интерактивный анализ данных с AI-комментариями
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple
import re
import os
import json


class DataScienceAI:
    """AI помощник для анализа данных"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.analysis_history = []
        
        # Ограничиваем размер данных для анализа
        max_rows = int(os.getenv('MAX_ROWS_ANALYSIS', '50000'))
        if len(df) > max_rows:
            self.df = df.sample(n=max_rows, random_state=42)
            print(f"⚠️ DataFrame sampled to {max_rows:,} rows for analysis")
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        Анализирует вопрос пользователя и возвращает ответ с графиками
        """
        question = question.lower().strip()
        
        # Определяем тип анализа
        analysis_type = self._detect_analysis_type(question)
        
        if analysis_type == "correlation":
            return self._analyze_correlations(question)
        elif analysis_type == "trend":
            return self._analyze_trends(question)
        elif analysis_type == "distribution":
            return self._analyze_distributions(question)
        elif analysis_type == "outliers":
            return self._analyze_outliers(question)
        elif analysis_type == "summary":
            return self._analyze_summary(question)
        elif analysis_type == "comparison":
            return self._analyze_comparisons(question)
        elif analysis_type == "prediction":
            return self._analyze_predictions(question)
        elif analysis_type == "clustering":
            return self._analyze_clustering(question)
        elif analysis_type == "statistical":
            return self._analyze_statistical_tests(question)
        else:
            return self._general_analysis(question)
    
    def _detect_analysis_type(self, question: str) -> str:
        """Определяет тип анализа по вопросу"""
        if any(word in question for word in ["корреляц", "correlation", "связь", "зависимость"]):
            return "correlation"
        elif any(word in question for word in ["тренд", "trend", "изменение", "динамика", "время"]):
            return "trend"
        elif any(word in question for word in ["распределение", "distribution", "гистограмма", "histogram"]):
            return "distribution"
        elif any(word in question for word in ["выброс", "outlier", "аномалия", "странн"]):
            return "outliers"
        elif any(word in question for word in ["сравн", "compare", "разница", "лучш", "худш"]):
            return "comparison"
        elif any(word in question for word in ["предсказ", "predict", "прогноз", "forecast", "модель", "model"]):
            return "prediction"
        elif any(word in question for word in ["кластер", "cluster", "групп", "сегмент", "segment"]):
            return "clustering"
        elif any(word in question for word in ["статистик", "statistical", "тест", "test", "значимость"]):
            return "statistical"
        elif any(word in question for word in ["общ", "summary", "итог", "статистика"]):
            return "summary"
        else:
            return "general"
    
    def _analyze_correlations(self, question: str) -> Dict[str, Any]:
        """Анализ корреляций между числовыми колонками"""
        numeric_cols = self.df.select_dtypes(include='number').columns
        
        if len(numeric_cols) < 2:
            return {
                "answer": "❌ Недостаточно числовых колонок для анализа корреляций",
                "charts": [],
                "insights": []
            }
        
        # Создаем корреляционную матрицу
        corr_matrix = self.df[numeric_cols].corr()
        
        # Находим сильные корреляции
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'col1': corr_matrix.columns[i],
                        'col2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        # Создаем heatmap
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Correlation Matrix",
                       color_continuous_scale="RdBu_r")
        fig.update_layout(
            plot_bgcolor='#131c2c',
            paper_bgcolor='#131c2c',
            font=dict(color='white')
        )
        
        # Формируем ответ
        answer = f"📊 **Анализ корреляций**\n\n"
        answer += f"Проанализировал {len(numeric_cols)} числовых колонок.\n\n"
        
        if strong_correlations:
            answer += "🔗 **Сильные корреляции (>0.7):**\n"
            for corr in strong_correlations[:5]:
                answer += f"• {corr['col1']} ↔ {corr['col2']}: {corr['correlation']:.3f}\n"
        else:
            answer += "ℹ️ Сильных корреляций не найдено (все < 0.7)\n"
        
        insights = [
            f"Найдено {len(strong_correlations)} сильных корреляций",
            f"Самые коррелирующие колонки: {', '.join(numeric_cols[:3])}",
            "Положительная корреляция означает, что значения растут вместе",
            "Отрицательная корреляция означает обратную зависимость"
        ]
        
        return {
            "answer": answer,
            "charts": [{
                "title": "Correlation Matrix",
                "html": self._create_chart_html(fig, f"chart_{hash(question)}_0")
            }],
            "insights": insights
        }
    
    def _analyze_trends(self, question: str) -> Dict[str, Any]:
        """Анализ трендов по времени"""
        # Ищем временные колонки
        time_cols = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    pd.to_datetime(self.df[col], errors='coerce')
                    if self.df[col].notna().sum() > len(self.df) * 0.5:
                        time_cols.append(col)
                except:
                    pass
        
        if not time_cols:
            return {
                "answer": "❌ Временные колонки не найдены для анализа трендов",
                "charts": [],
                "insights": []
            }
        
        time_col = time_cols[0]
        numeric_cols = self.df.select_dtypes(include='number').columns
        
        if len(numeric_cols) == 0:
            return {
                "answer": "❌ Нет числовых колонок для анализа трендов",
                "charts": [],
                "insights": []
            }
        
        # Создаем трендовые графики
        charts = []
        insights = []
        
        for col in numeric_cols[:3]:  # Первые 3 числовые колонки
            try:
                df_clean = self.df.dropna(subset=[time_col, col])
                df_clean[time_col] = pd.to_datetime(df_clean[time_col], errors='coerce')
                df_clean = df_clean.sort_values(time_col)
                
                fig = px.line(df_clean, x=time_col, y=col, 
                            title=f"Trend Analysis: {col}",
                            markers=True)
                fig.update_layout(
                    plot_bgcolor='#131c2c',
                    paper_bgcolor='#131c2c',
                    font=dict(color='white')
                )
                
                charts.append({
                    "title": f"Trend: {col}",
                    "html": self._create_chart_html(fig, f"chart_{hash(question)}_0")
                })
                
                # Анализируем тренд
                if len(df_clean) > 1:
                    first_val = df_clean[col].iloc[0]
                    last_val = df_clean[col].iloc[-1]
                    change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
                    
                    trend_direction = "📈 растет" if change > 0 else "📉 падает" if change < 0 else "➡️ стабилен"
                    insights.append(f"{col}: {trend_direction} на {abs(change):.1f}%")
                
            except Exception as e:
                continue
        
        answer = f"📈 **Анализ трендов**\n\n"
        answer += f"Проанализировал тренды по колонке времени: {time_col}\n"
        answer += f"Создал {len(charts)} графиков трендов\n\n"
        
        if insights:
            answer += "**Основные наблюдения:**\n"
            for insight in insights[:5]:
                answer += f"• {insight}\n"
        
        return {
            "answer": answer,
            "charts": charts,
            "insights": insights
        }
    
    def _analyze_distributions(self, question: str) -> Dict[str, Any]:
        """Анализ распределений данных"""
        numeric_cols = self.df.select_dtypes(include='number').columns
        
        if len(numeric_cols) == 0:
            return {
                "answer": "❌ Нет числовых колонок для анализа распределений",
                "charts": [],
                "insights": []
            }
        
        charts = []
        insights = []
        
        for col in numeric_cols[:3]:  # Первые 3 числовые колонки
            try:
                # Гистограмма
                fig = px.histogram(self.df, x=col, 
                                 title=f"Distribution: {col}",
                                 nbins=30)
                fig.update_layout(
                    plot_bgcolor='#131c2c',
                    paper_bgcolor='#131c2c',
                    font=dict(color='white')
                )
                
                charts.append({
                    "title": f"Distribution: {col}",
                    "html": self._create_chart_html(fig, f"chart_{hash(question)}_0")
                })
                
                # Анализируем распределение
                stats = self.df[col].describe()
                skewness = self.df[col].skew()
                
                if abs(skewness) < 0.5:
                    shape = "нормальное"
                elif skewness > 0.5:
                    shape = "правостороннее (положительная асимметрия)"
                else:
                    shape = "левостороннее (отрицательная асимметрия)"
                
                insights.append(f"{col}: {shape} распределение, среднее: {stats['mean']:.2f}")
                
            except Exception as e:
                continue
        
        answer = f"📊 **Анализ распределений**\n\n"
        answer += f"Проанализировал распределения {len(numeric_cols)} числовых колонок\n"
        answer += f"Создал {len(charts)} гистограмм\n\n"
        
        if insights:
            answer += "**Характеристики распределений:**\n"
            for insight in insights[:5]:
                answer += f"• {insight}\n"
        
        return {
            "answer": answer,
            "charts": charts,
            "insights": insights
        }
    
    def _analyze_outliers(self, question: str) -> Dict[str, Any]:
        """Анализ выбросов в данных"""
        numeric_cols = self.df.select_dtypes(include='number').columns
        
        if len(numeric_cols) == 0:
            return {
                "answer": "❌ Нет числовых колонок для анализа выбросов",
                "charts": [],
                "insights": []
            }
        
        charts = []
        insights = []
        outlier_data = []
        
        for col in numeric_cols[:3]:  # Первые 3 числовые колонки
            try:
                # Box plot для выбросов
                fig = px.box(self.df, y=col, title=f"Outliers: {col}")
                fig.update_layout(
                    plot_bgcolor='#131c2c',
                    paper_bgcolor='#131c2c',
                    font=dict(color='white')
                )
                
                charts.append({
                    "title": f"Outliers: {col}",
                    "html": self._create_chart_html(fig, f"chart_{hash(question)}_0")
                })
                
                # Находим выбросы по IQR
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_percent = (outlier_count / len(self.df)) * 100
                
                outlier_data.append({
                    'column': col,
                    'count': outlier_count,
                    'percent': outlier_percent
                })
                
                if outlier_count > 0:
                    insights.append(f"{col}: {outlier_count} выбросов ({outlier_percent:.1f}%)")
                else:
                    insights.append(f"{col}: выбросов не найдено")
                
            except Exception as e:
                continue
        
        answer = f"🔍 **Анализ выбросов**\n\n"
        answer += f"Проанализировал {len(numeric_cols)} числовых колонок на наличие выбросов\n"
        answer += f"Создал {len(charts)} box plots\n\n"
        
        if outlier_data:
            answer += "**Найденные выбросы:**\n"
            for data in outlier_data:
                if data['count'] > 0:
                    answer += f"• {data['column']}: {data['count']} выбросов ({data['percent']:.1f}%)\n"
                else:
                    answer += f"• {data['column']}: выбросов нет\n"
        
        return {
            "answer": answer,
            "charts": charts,
            "insights": insights
        }
    
    def _analyze_summary(self, question: str) -> Dict[str, Any]:
        """Общий анализ данных"""
        numeric_cols = self.df.select_dtypes(include='number').columns
        categorical_cols = self.df.select_dtypes(include='object').columns
        
        answer = f"📋 **Общий анализ данных**\n\n"
        answer += f"**Размер данных:** {len(self.df)} строк, {len(self.df.columns)} колонок\n"
        answer += f"**Числовые колонки:** {len(numeric_cols)}\n"
        answer += f"**Категориальные колонки:** {len(categorical_cols)}\n\n"
        
        # Пропущенные значения
        missing_data = self.df.isnull().sum()
        missing_cols = missing_data[missing_data > 0]
        
        if len(missing_cols) > 0:
            answer += f"**Пропущенные значения:** {len(missing_cols)} колонок\n"
            for col, count in missing_cols.head(3).items():
                percent = (count / len(self.df)) * 100
                answer += f"• {col}: {count} ({percent:.1f}%)\n"
        else:
            answer += "**Пропущенные значения:** отсутствуют\n"
        
        answer += "\n**Рекомендации:**\n"
        if len(numeric_cols) > 0:
            answer += "• Проанализируйте корреляции между числовыми колонками\n"
        if len(categorical_cols) > 0:
            answer += "• Изучите распределение по категориям\n"
        if len(missing_cols) > 0:
            answer += "• Обработайте пропущенные значения\n"
        
        # Добавляем простой график для тестирования
        charts = []
        if len(numeric_cols) > 0:
            try:
                col = numeric_cols[0]
                data = self.df[col].dropna()
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=data,
                    name=f'Distribution of {col}',
                    nbinsx=20
                ))
                
                fig.update_layout(
                    title=f"Data Overview: {col}",
                    xaxis_title=col,
                    yaxis_title="Frequency",
                    plot_bgcolor='#131c2c',
                    paper_bgcolor='#131c2c',
                    font=dict(color='white'),
                    height=400
                )
                
                charts.append({
                    "title": f"Data Overview: {col}",
                    "html": fig.to_html(full_html=False, include_plotlyjs=True, div_id=f"summary_chart_{hash(question)}")
                })
            except Exception as e:
                print(f"Error creating summary chart: {e}")
        
        return {
            "answer": answer,
            "charts": charts,
            "insights": [
                f"Данные содержат {len(numeric_cols)} числовых и {len(categorical_cols)} категориальных колонок",
                f"Общий размер: {len(self.df)} записей",
                "Рекомендуется начать с анализа корреляций и распределений"
            ]
        }
    
    def _analyze_comparisons(self, question: str) -> Dict[str, Any]:
        """Сравнительный анализ"""
        categorical_cols = self.df.select_dtypes(include='object').columns
        numeric_cols = self.df.select_dtypes(include='number').columns
        
        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            return {
                "answer": "❌ Недостаточно данных для сравнительного анализа",
                "charts": [],
                "insights": []
            }
        
        charts = []
        insights = []
        
        # Сравниваем по категориям
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        
        try:
            # Box plot для сравнения
            fig = px.box(self.df, x=cat_col, y=num_col, 
                        title=f"Comparison: {num_col} by {cat_col}")
            fig.update_layout(
                plot_bgcolor='#131c2c',
                paper_bgcolor='#131c2c',
                font=dict(color='white'),
                xaxis_tickangle=-45
            )
            
            charts.append({
                "title": f"Comparison: {num_col} by {cat_col}",
                "html": self._create_chart_html(fig, f"chart_{hash(question)}_0")
            })
            
            # Статистика по категориям
            category_stats = self.df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std']).round(2)
            
            insights.append(f"Сравнение {num_col} по {cat_col}:")
            for category, stats in category_stats.head(5).iterrows():
                insights.append(f"• {category}: среднее={stats['mean']:.2f}, медиана={stats['median']:.2f}")
            
        except Exception as e:
            pass
        
        answer = f"⚖️ **Сравнительный анализ**\n\n"
        answer += f"Сравниваю {num_col} по категориям {cat_col}\n"
        answer += f"Создал {len(charts)} графиков сравнения\n\n"
        
        if insights:
            answer += "**Результаты сравнения:**\n"
            for insight in insights[:5]:
                answer += f"{insight}\n"
        
        return {
            "answer": answer,
            "charts": charts,
            "insights": insights
        }
    
    def _general_analysis(self, question: str) -> Dict[str, Any]:
        """Общий анализ на основе вопроса"""
        answer = f"🤖 **AI Анализ**\n\n"
        answer += f"Проанализировал ваш вопрос: \"{question}\"\n\n"
        
        # Базовые метрики
        numeric_cols = self.df.select_dtypes(include='number').columns
        categorical_cols = self.df.select_dtypes(include='object').columns
        
        answer += f"**Данные для анализа:**\n"
        answer += f"• {len(self.df)} записей\n"
        answer += f"• {len(numeric_cols)} числовых колонок: {', '.join(numeric_cols[:3])}\n"
        answer += f"• {len(categorical_cols)} категориальных колонок: {', '.join(categorical_cols[:3])}\n\n"
        
        answer += "**Рекомендации для анализа:**\n"
        if len(numeric_cols) > 1:
            answer += "• Спросите о корреляциях между числовыми колонками\n"
        if len(categorical_cols) > 0:
            answer += "• Изучите распределение по категориям\n"
        if len(numeric_cols) > 0:
            answer += "• Проанализируйте тренды и выбросы\n"
        
        return {
            "answer": answer,
            "charts": [],
            "insights": [
                "Используйте конкретные вопросы для получения детального анализа",
                "Попробуйте спросить о корреляциях, трендах или распределениях"
            ]
        }
    
    def _analyze_predictions(self, question: str) -> Dict[str, Any]:
        """ML предсказания и прогнозы"""
        numeric_cols = self.df.select_dtypes(include='number').columns
        
        if len(numeric_cols) < 2:
            return {
                "answer": "❌ Недостаточно числовых колонок для предсказаний",
                "charts": [],
                "insights": []
            }
        
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_squared_error
            import numpy as np
            
            # Простая линейная регрессия
            X_col = numeric_cols[0]
            y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            
            # Подготавливаем данные
            df_clean = self.df[[X_col, y_col]].dropna()
            if len(df_clean) < 10:
                return {
                    "answer": "❌ Недостаточно данных для построения модели (минимум 10 записей)",
                    "charts": [],
                    "insights": []
                }
            
            X = df_clean[[X_col]].values
            y = df_clean[y_col].values
            
            # Разделяем на train/test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Обучаем модель
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Предсказания
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Создаем график
            fig = go.Figure()
            
            # Точки данных
            fig.add_trace(go.Scatter(
                x=df_clean[X_col], 
                y=df_clean[y_col], 
                mode='markers',
                name='Actual Data',
                marker=dict(color='blue', size=8)
            ))
            
            # Линия регрессии
            x_range = np.linspace(df_clean[X_col].min(), df_clean[X_col].max(), 100)
            y_range = model.predict(x_range.reshape(-1, 1))
            
            fig.add_trace(go.Scatter(
                x=x_range, 
                y=y_range, 
                mode='lines',
                name='Prediction Line',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title=f"Prediction Model: {y_col} vs {X_col}",
                xaxis_title=X_col,
                yaxis_title=y_col,
                plot_bgcolor='#131c2c',
                paper_bgcolor='#131c2c',
                font=dict(color='white')
            )
            
            # Формируем ответ
            answer = f"🔮 **ML Предсказательная модель**\n\n"
            answer += f"**Модель:** Линейная регрессия\n"
            answer += f"**Целевая переменная:** {y_col}\n"
            answer += f"**Признак:** {X_col}\n"
            answer += f"**Точность (R²):** {r2:.3f}\n"
            answer += f"**Средняя ошибка (MSE):** {mse:.3f}\n\n"
            
            if r2 > 0.7:
                answer += "✅ **Отличная модель** - высокая точность предсказаний\n"
            elif r2 > 0.5:
                answer += "⚠️ **Удовлетворительная модель** - средняя точность\n"
            else:
                answer += "❌ **Слабая модель** - низкая точность, нужны дополнительные признаки\n"
            
            answer += f"\n**Уравнение:** {y_col} = {model.coef_[0]:.3f} × {X_col} + {model.intercept_:.3f}\n"
            
            insights = [
                f"Модель объясняет {r2*100:.1f}% вариации данных",
                f"Средняя ошибка предсказания: {np.sqrt(mse):.2f}",
                "Линейная регрессия показывает простую зависимость между переменными",
                "Для улучшения точности можно добавить больше признаков"
            ]
            
            return {
                "answer": answer,
                "charts": [{
                    "title": f"Prediction Model: {y_col} vs {X_col}",
                    "html": fig.to_html(full_html=False, include_plotlyjs=True, div_id=f"prediction_chart_{hash(question)}")
                }],
                "insights": insights
            }
            
        except ImportError:
            return {
                "answer": "❌ Для ML анализа требуется scikit-learn. Установите: pip install scikit-learn",
                "charts": [],
                "insights": []
            }
        except Exception as e:
            return {
                "answer": f"❌ Ошибка при построении модели: {str(e)}",
                "charts": [],
                "insights": []
            }
    
    def _analyze_clustering(self, question: str) -> Dict[str, Any]:
        """Кластеризация и сегментация данных"""
        numeric_cols = self.df.select_dtypes(include='number').columns
        
        if len(numeric_cols) < 2:
            return {
                "answer": "❌ Недостаточно числовых колонок для кластеризации",
                "charts": [],
                "insights": []
            }
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            # Выбираем первые 2 числовые колонки для визуализации
            X_col = numeric_cols[0]
            y_col = numeric_cols[1]
            
            # Подготавливаем данные
            df_clean = self.df[[X_col, y_col]].dropna()
            if len(df_clean) < 10:
                return {
                    "answer": "❌ Недостаточно данных для кластеризации (минимум 10 записей)",
                    "charts": [],
                    "insights": []
                }
            
            # Нормализуем данные
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_clean[[X_col, y_col]])
            
            # Определяем оптимальное количество кластеров (метод локтя)
            max_clusters = min(8, len(df_clean) // 2)
            if max_clusters < 2:
                max_clusters = 2
            
            inertias = []
            for k in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            
            # Выбираем оптимальное количество кластеров (простая эвристика)
            optimal_k = 3  # По умолчанию
            if len(inertias) > 2:
                # Ищем "локоть" в кривой
                diffs = np.diff(inertias)
                if len(diffs) > 1:
                    optimal_k = np.argmin(diffs) + 2
            
            # Финальная кластеризация
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Создаем график
            fig = go.Figure()
            
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for i in range(optimal_k):
                mask = clusters == i
                fig.add_trace(go.Scatter(
                    x=df_clean[mask][X_col],
                    y=df_clean[mask][y_col],
                    mode='markers',
                    name=f'Cluster {i+1}',
                    marker=dict(color=colors[i % len(colors)], size=8)
                ))
            
            # Центроиды
            centroids = scaler.inverse_transform(kmeans.cluster_centers_)
            fig.add_trace(go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode='markers',
                name='Centroids',
                marker=dict(color='black', size=15, symbol='x')
            ))
            
            fig.update_layout(
                title=f"Clustering Analysis: {X_col} vs {y_col}",
                xaxis_title=X_col,
                yaxis_title=y_col,
                plot_bgcolor='#131c2c',
                paper_bgcolor='#131c2c',
                font=dict(color='white')
            )
            
            # Анализируем кластеры
            cluster_stats = []
            for i in range(optimal_k):
                mask = clusters == i
                cluster_data = df_clean[mask]
                cluster_stats.append({
                    'cluster': i+1,
                    'size': len(cluster_data),
                    'x_mean': cluster_data[X_col].mean(),
                    'y_mean': cluster_data[y_col].mean()
                })
            
            # Формируем ответ
            answer = f"🎯 **Кластерный анализ**\n\n"
            answer += f"**Количество кластеров:** {optimal_k}\n"
            answer += f"**Общее количество точек:** {len(df_clean)}\n\n"
            
            answer += "**Характеристики кластеров:**\n"
            for stat in cluster_stats:
                answer += f"• **Кластер {stat['cluster']}:** {stat['size']} точек, "
                answer += f"центр ({stat['x_mean']:.2f}, {stat['y_mean']:.2f})\n"
            
            insights = [
                f"Данные разделены на {optimal_k} логических группы",
                f"Самый большой кластер содержит {max(stat['size'] for stat in cluster_stats)} точек",
                "Кластеры показывают естественные группировки в данных",
                "Можно использовать для сегментации клиентов или продуктов"
            ]
            
            return {
                "answer": answer,
                "charts": [{
                    "title": f"Clustering: {X_col} vs {y_col}",
                    "html": fig.to_html(full_html=False, include_plotlyjs=True, div_id=f"clustering_chart_{hash(question)}")
                }],
                "insights": insights
            }
            
        except ImportError:
            return {
                "answer": "❌ Для кластеризации требуется scikit-learn. Установите: pip install scikit-learn",
                "charts": [],
                "insights": []
            }
        except Exception as e:
            return {
                "answer": f"❌ Ошибка при кластеризации: {str(e)}",
                "charts": [],
                "insights": []
            }
    
    def _analyze_statistical_tests(self, question: str) -> Dict[str, Any]:
        """Статистические тесты и значимость"""
        numeric_cols = self.df.select_dtypes(include='number').columns
        categorical_cols = self.df.select_dtypes(include='object').columns
        
        if len(numeric_cols) == 0:
            return {
                "answer": "❌ Нет числовых колонок для статистических тестов",
                "charts": [],
                "insights": []
            }
        
        try:
            from scipy import stats
            import numpy as np
            
            results = []
            charts = []
            
            # 1. Тест на нормальность (Shapiro-Wilk)
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                data = self.df[col].dropna()
                
                if len(data) >= 3 and len(data) <= 5000:  # Ограничения теста
                    stat, p_value = stats.shapiro(data)
                    results.append(f"**Тест нормальности ({col}):** p-value = {p_value:.4f}")
                    
                    if p_value > 0.05:
                        results.append(f"✅ Данные {col} распределены нормально (p > 0.05)")
                    else:
                        results.append(f"❌ Данные {col} НЕ нормальны (p ≤ 0.05)")
            
            # 2. T-test для сравнения групп
            if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                num_col = numeric_cols[0]
                cat_col = categorical_cols[0]
                
                # Берем только первые 2 категории
                unique_cats = self.df[cat_col].value_counts().head(2).index
                if len(unique_cats) >= 2:
                    group1 = self.df[self.df[cat_col] == unique_cats[0]][num_col].dropna()
                    group2 = self.df[self.df[cat_col] == unique_cats[1]][num_col].dropna()
                    
                    if len(group1) >= 2 and len(group2) >= 2:
                        stat, p_value = stats.ttest_ind(group1, group2)
                        results.append(f"**T-test ({num_col} по {cat_col}):** p-value = {p_value:.4f}")
                        
                        if p_value < 0.05:
                            results.append(f"✅ Значимая разница между группами (p < 0.05)")
                        else:
                            results.append(f"❌ Нет значимой разности (p ≥ 0.05)")
            
            # 3. Корреляционный тест
            if len(numeric_cols) >= 2:
                col1, col2 = numeric_cols[0], numeric_cols[1]
                data1 = self.df[col1].dropna()
                data2 = self.df[col2].dropna()
                
                # Находим общие индексы
                common_idx = data1.index.intersection(data2.index)
                if len(common_idx) >= 3:
                    corr, p_value = stats.pearsonr(data1[common_idx], data2[common_idx])
                    results.append(f"**Корреляционный тест ({col1} vs {col2}):** r = {corr:.3f}, p = {p_value:.4f}")
                    
                    if p_value < 0.05:
                        results.append(f"✅ Значимая корреляция (p < 0.05)")
                    else:
                        results.append(f"❌ Корреляция не значима (p ≥ 0.05)")
            
            # Создаем график распределения
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                data = self.df[col].dropna()
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=data,
                    name=f'Distribution of {col}',
                    nbinsx=30
                ))
                
                # Добавляем нормальное распределение для сравнения
                mu, sigma = data.mean(), data.std()
                x_range = np.linspace(data.min(), data.max(), 100)
                normal_dist = stats.norm.pdf(x_range, mu, sigma) * len(data) * (data.max() - data.min()) / 30
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=normal_dist,
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title=f"Distribution Analysis: {col}",
                    xaxis_title=col,
                    yaxis_title="Frequency",
                    plot_bgcolor='#131c2c',
                    paper_bgcolor='#131c2c',
                    font=dict(color='white')
                )
                
                charts.append({
                    "title": f"Distribution: {col}",
                    "html": self._create_chart_html(fig, f"chart_{hash(question)}_0")
                })
            
            # Формируем ответ
            answer = f"📊 **Статистические тесты**\n\n"
            answer += f"**Проанализировано:** {len(numeric_cols)} числовых колонок\n"
            answer += f"**Выполнено тестов:** {len(results)//2}\n\n"
            
            for result in results:
                answer += f"{result}\n"
            
            insights = [
                "Статистические тесты помогают проверить гипотезы о данных",
                "p-value < 0.05 означает статистическую значимость",
                "Нормальное распределение важно для многих тестов",
                "Корреляция не означает причинно-следственную связь"
            ]
            
            return {
                "answer": answer,
                "charts": charts,
                "insights": insights
            }
            
        except ImportError:
            return {
                "answer": "❌ Для статистических тестов требуется scipy. Установите: pip install scipy",
                "charts": [],
                "insights": []
            }
    
    def _create_chart_html(self, fig, div_id):
        """Создает HTML для графика"""
        # Просто генерируем HTML с Plotly.js - это самый надежный способ
        return fig.to_html(full_html=False, include_plotlyjs=True, div_id=div_id)

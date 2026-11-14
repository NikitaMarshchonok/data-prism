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
import numpy as np
from src.column_semantics import profile_dataframe_columns
from src.structural_dependencies import additive_pair_dependency, detect_additive_dependencies

from .insight_engine import EvidenceBasedInsightEngine
from .statistical_engine import StatisticalValidationEngine
from .anomaly_segmentation_engine import AnomalySegmentationEngine


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
        
        # Получаем основной анализ
        if analysis_type == "correlation":
            result = self._analyze_correlations(question)
        elif analysis_type == "trend":
            result = self._analyze_trends(question)
        elif analysis_type == "distribution":
            result = self._analyze_distributions(question)
        elif analysis_type == "outliers":
            result = self._analyze_outliers(question)
        elif analysis_type == "summary":
            result = self._analyze_summary(question)
        elif analysis_type == "comparison":
            result = self._analyze_comparisons(question)
        elif analysis_type == "prediction":
            result = self._analyze_predictions(question)
        elif analysis_type == "clustering":
            result = self._analyze_clustering(question)
        elif analysis_type == "statistical":
            result = self._analyze_statistical_tests(question)
        else:
            result = self._general_analysis(question)

        # Добавляем рассчитанные выводы с проверяемыми доказательствами.
        if "insights" not in result:
            result["insights"] = []
        result["insights"].extend(EvidenceBasedInsightEngine(self.df).generate())

        return result
    
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
                "answer": "❌ Correlation analysis requires at least two numeric columns.",
                "charts": [],
                "insights": []
            }
        
        # Создаем корреляционную матрицу
        corr_matrix = self.df[numeric_cols].corr()
        dependencies = detect_additive_dependencies(self.df, profile_dataframe_columns(self.df))
        
        # Находим сильные корреляции
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if additive_pair_dependency(
                    [str(corr_matrix.columns[i]), str(corr_matrix.columns[j])], dependencies
                ):
                    continue
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
        answer = f"📊 **Correlation analysis**\n\n"
        answer += f"Numeric columns analyzed: {len(numeric_cols)}.\n\n"
        
        if strong_correlations:
            answer += "🔗 **Strong correlations (absolute r > 0.7):**\n"
            for corr in strong_correlations[:5]:
                answer += f"• {corr['col1']} ↔ {corr['col2']}: {corr['correlation']:.3f}\n"
        else:
            answer += "ℹ️ No strong correlations among the remaining pairs.\n"

        if dependencies:
            answer += "\nArithmetic relationships in the rows analyzed:\n"
            for dependency in dependencies:
                answer += f"• {dependency['equation']} — {dependency['evaluated_rows']} rows.\n"
            answer += (
                "The total and its components share a built-in arithmetic relationship. "
                "The matrix includes these coefficients for inspection. Before modeling, "
                "confirm the column definitions and check when the components become available.\n"
            )
        
        insights = [
            f"Strong correlations outside detected total-component relationships: {len(strong_correlations)}",
            "With positive correlation, higher values of one variable tend to accompany higher values of the other.",
            "With negative correlation, higher values of one variable tend to accompany lower values of the other."
        ]
        
        return {
            "answer": answer,
            "charts": [{
                "title": "Correlation Matrix",
                "html": self._create_chart_html(fig, f"chart_{hash(question)}_0", "Correlation Matrix")
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
                "answer": "❌ Trend analysis requires a time column.",
                "charts": [],
                "insights": []
            }
        
        time_col = time_cols[0]
        numeric_cols = self.df.select_dtypes(include='number').columns
        
        if len(numeric_cols) == 0:
            return {
                "answer": "❌ Trend analysis requires a numeric column.",
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
                    "html": self._create_chart_html(fig, f"chart_{hash(question)}_0", f"Trend Analysis: {col}")
                })
                
                # Анализируем тренд
                if len(df_clean) > 1:
                    first_val = df_clean[col].iloc[0]
                    last_val = df_clean[col].iloc[-1]
                    change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
                    
                    trend_direction = "📈 increased" if change > 0 else "📉 decreased" if change < 0 else "➡️ unchanged"
                    insights.append(f"{col}: {trend_direction} ({abs(change):.1f}% change)")
                
            except Exception as e:
                continue
        
        answer = f"📈 **Trend analysis**\n\n"
        answer += f"Time column: {time_col}\n"
        answer += f"Trend charts: {len(charts)}\n\n"
        
        if insights:
            answer += "**Key observations:**\n"
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
                "answer": "❌ Distribution analysis requires a numeric column.",
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
                    "html": self._create_chart_html(fig, f"chart_{hash(question)}_0", f"Distribution Analysis: {col}")
                })
                
                # Анализируем распределение
                stats = self.df[col].describe()
                skewness = self.df[col].skew()
                
                if abs(skewness) < 0.5:
                    shape = "roughly symmetric"
                elif skewness > 0.5:
                    shape = "right-skewed"
                else:
                    shape = "left-skewed"
                
                insights.append(f"{col}: {shape} distribution, mean: {stats['mean']:.2f}")
                
            except Exception as e:
                continue
        
        answer = f"📊 **Distribution analysis**\n\n"
        answer += f"Numeric columns reviewed: {len(numeric_cols)}\n"
        answer += f"Histograms: {len(charts)}\n\n"
        
        if insights:
            answer += "**Distribution summaries:**\n"
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
                "answer": "❌ Outlier analysis requires a numeric column.",
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
                    "html": self._create_chart_html(fig, f"chart_{hash(question)}_0", f"Outlier Detection: {col}")
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
                    insights.append(f"{col}: {outlier_count} potential outliers ({outlier_percent:.1f}%)")
                else:
                    insights.append(f"{col}: no potential outliers found")
                
            except Exception as e:
                continue
        
        answer = f"🔍 **Outlier analysis**\n\n"
        answer += f"Numeric columns reviewed for outliers: {len(numeric_cols)}\n"
        answer += f"Box plots: {len(charts)}\n\n"
        
        if outlier_data:
            answer += "**Potential outliers:**\n"
            for data in outlier_data:
                if data['count'] > 0:
                    answer += f"• {data['column']}: {data['count']} potential outliers ({data['percent']:.1f}%)\n"
                else:
                    answer += f"• {data['column']}: no potential outliers found\n"
        
        return {
            "answer": answer,
            "charts": charts,
            "insights": insights
        }
    
    def _analyze_summary(self, question: str) -> Dict[str, Any]:
        """Общий анализ данных"""
        numeric_cols = self.df.select_dtypes(include='number').columns
        categorical_cols = self.df.select_dtypes(
            include=["object", "string", "category", "bool"]
        ).columns
        
        answer = f"📋 **Dataset overview**\n\n"
        answer += f"**Dataset size:** {len(self.df)} rows, {len(self.df.columns)} columns\n"
        answer += f"**Numeric columns:** {len(numeric_cols)}\n"
        answer += f"**Categorical columns:** {len(categorical_cols)}\n\n"
        
        # Пропущенные значения
        missing_data = self.df.isnull().sum()
        missing_cols = missing_data[missing_data > 0]
        
        if len(missing_cols) > 0:
            answer += f"**Missing values:** {len(missing_cols)} affected columns\n"
            for col, count in missing_cols.head(3).items():
                percent = (count / len(self.df)) * 100
                answer += f"• {col}: {count} ({percent:.1f}%)\n"
        else:
            answer += "**Missing values:** None\n"
        
        answer += "\n**Recommendations:**\n"
        if len(numeric_cols) > 0:
            answer += "• Review correlations between numeric columns.\n"
        if len(categorical_cols) > 0:
            answer += "• Review the frequency of each category.\n"
        if len(missing_cols) > 0:
            answer += "• Review missing values and decide how to handle them.\n"
        
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
                f"The dataset has {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns.",
                f"Total rows: {len(self.df)}",
                "Start with correlations and distributions."
            ]
        }
    
    def _analyze_comparisons(self, question: str) -> Dict[str, Any]:
        """Сравнительный анализ"""
        categorical_cols = self.df.select_dtypes(
            include=["object", "string", "category", "bool"]
        ).columns
        numeric_cols = self.df.select_dtypes(include='number').columns
        
        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            return {
                "answer": "❌ This comparison requires a numeric column and a categorical column.",
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
                "html": self._create_chart_html(fig, f"chart_{hash(question)}_0", f"Comparison: {num_col} by {cat_col}")
            })
            
            # Статистика по категориям
            category_stats = self.df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std']).round(2)
            
            insights.append(f"Comparing {num_col} by {cat_col}:")
            for category, stats in category_stats.head(5).iterrows():
                insights.append(f"• {category}: mean={stats['mean']:.2f}, median={stats['median']:.2f}")
            
        except Exception as e:
            pass
        
        answer = f"⚖️ **Comparison**\n\n"
        answer += f"Comparing {num_col} across {cat_col} categories.\n"
        answer += f"Comparison charts: {len(charts)}\n\n"
        
        if insights:
            answer += "**Comparison results:**\n"
            for insight in insights[:5]:
                answer += f"{insight}\n"
        
        return {
            "answer": answer,
            "charts": charts,
            "insights": insights
        }
    
    def _general_analysis(self, question: str) -> Dict[str, Any]:
        """Общий анализ на основе вопроса"""
        answer = f"🤖 **Data analysis**\n\n"
        answer += f"Your question: \"{question}\"\n\n"
        
        # Базовые метрики
        numeric_cols = self.df.select_dtypes(include='number').columns
        categorical_cols = self.df.select_dtypes(
            include=["object", "string", "category", "bool"]
        ).columns
        
        answer += f"**Available data:**\n"
        answer += f"• {len(self.df)} rows\n"
        answer += f"• {len(numeric_cols)} numeric columns: {', '.join(numeric_cols[:3])}\n"
        answer += f"• {len(categorical_cols)} categorical columns: {', '.join(categorical_cols[:3])}\n\n"
        
        answer += "**Suggested analyses:**\n"
        if len(numeric_cols) > 1:
            answer += "• Ask about correlations between numeric columns.\n"
        if len(categorical_cols) > 0:
            answer += "• Review the frequency of each category.\n"
        if len(numeric_cols) > 0:
            answer += "• Explore trends and potential outliers.\n"
        
        return {
            "answer": answer,
            "charts": [],
            "insights": [
                "Name the columns and the analysis you want.",
                "Try asking about correlations, trends, or distributions."
            ]
        }
    
    def _analyze_predictions(self, question: str) -> Dict[str, Any]:
        """ML предсказания и прогнозы"""
        numeric_cols = self.df.select_dtypes(include='number').columns
        
        if len(numeric_cols) < 2:
            return {
                "answer": "❌ This model requires at least two numeric columns.",
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
                    "answer": "❌ Model training requires at least 10 usable rows.",
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
            answer = f"🔮 **Predictive model**\n\n"
            answer += f"**Model:** Linear regression\n"
            answer += f"**Target:** {y_col}\n"
            answer += f"**Feature:** {X_col}\n"
            answer += f"**Test-set R²:** {r2:.3f}\n"
            answer += f"**Test-set mean squared error (MSE):** {mse:.3f}\n\n"
            
            if r2 > 0.7:
                answer += "✅ **High test-set R²**\n"
            elif r2 > 0.5:
                answer += "⚠️ **Moderate test-set R²**\n"
            else:
                answer += "❌ **Low test-set R²**\n"
            
            answer += f"\n**Equation:** {y_col} = {model.coef_[0]:.3f} × {X_col} + {model.intercept_:.3f}\n"
            
            insights = [
                f"Test-set R²: {r2:.3f}",
                f"Test-set root mean squared error: {np.sqrt(mse):.2f}",
                "Linear regression estimates a linear relationship between the two variables.",
                "Additional available features can be evaluated on held-out data."
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
                "answer": "❌ The modeling service requires scikit-learn.",
                "charts": [],
                "insights": []
            }
        except Exception as e:
            return {
                "answer": f"❌ Could not fit the model: {str(e)}",
                "charts": [],
                "insights": []
            }
    
    def _analyze_clustering(self, question: str) -> Dict[str, Any]:
        """Return validated multivariate anomalies and exploratory segments."""
        engine = AnomalySegmentationEngine(self.df)
        report = engine.analyze()
        if report["status"] != "ok":
            return {
                "answer": f"🧭 **Anomalies & Segments**\n\n{report['summary']}",
                "charts": [],
                "insights": [],
                "pattern_analysis": report,
            }

        anomaly = report["anomaly_detection"]
        segmentation = report["segmentation"]
        answer_lines = [
            "🧭 **Anomalies & Segments**",
            "",
            anomaly["summary"],
        ]
        if anomaly["top_anomalies"]:
            answer_lines.extend(["", "**Highest-priority rows:**"])
            for item in anomaly["top_anomalies"][:5]:
                reason = item["reasons"][0]
                answer_lines.append(
                    f"• Row {item['row_index']}: score={item['anomaly_score']:.3f}; "
                    f"{reason['feature']} robust deviation={reason['robust_deviation']:+.2f}"
                )

        answer_lines.extend(["", segmentation["summary"]])
        if segmentation.get("status") == "ok":
            answer_lines.append("**Segment profiles:**")
            for segment in segmentation["segments"]:
                leading = segment["distinguishing_features"][0]
                answer_lines.append(
                    f"• Segment {segment['segment']}: {segment['size']} rows "
                    f"({segment['share']:.1%}); strongest distinction: "
                    f"{leading['feature']} ({leading['robust_difference']:+.2f})"
                )

        answer_lines.extend(
            [
                "",
                "These are review candidates and exploratory segments, not causal or operational conclusions.",
            ]
        )
        return {
            "answer": "\n".join(answer_lines),
            "charts": [],
            "insights": engine.evidence_insights(report),
            "pattern_analysis": report,
        }

    def _analyze_statistical_tests(self, question: str) -> Dict[str, Any]:
        """Run auditable tests with effect sizes, intervals and FDR control."""
        engine = StatisticalValidationEngine(self.df)
        report = engine.analyze()
        if not report["tests"]:
            return {
                "answer": f"🧪 **Statistical Validation**\n\n{report['summary']}",
                "charts": [],
                "insights": [],
                "statistical_validation": report,
            }

        answer_lines = [
            "🧪 **Statistical Validation**",
            "",
            report["summary"],
            f"Correction: {report['correction']}.",
            "",
        ]
        for test in report["tests"]:
            decision = (
                "observed arithmetic relationship"
                if test.get("structural_dependency")
                else "FDR significant" if test["significant"] else "inconclusive"
            )
            answer_lines.extend(
                [
                    f"**{test['title']}** — {decision}",
                    test["interpretation"],
                    (
                        f"Method: {test['method']}; n={test['sample_size']}; "
                        f"raw p={test['raw_p_value']:.4g}; "
                        f"adjusted p={test['adjusted_p_value']:.4g}."
                    ),
                    "",
                ]
            )

        return {
            "answer": "\n".join(answer_lines),
            "charts": [],
            "insights": engine.evidence_insights(report),
            "statistical_validation": report,
        }

    def _create_chart_html(self, fig, div_id, chart_title=""):
        """Создает HTML графика без недоказуемых шаблонных комментариев."""
        return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)

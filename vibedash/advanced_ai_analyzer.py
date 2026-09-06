"""
Advanced AI Analyzer for Data Science Charts and Visualizations
Интегрирует продвинутые AI модели для профессионального анализа графиков
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple
import json
import os
import numpy as np
from datetime import datetime

class AdvancedAIAnalyzer:
    """
    Продвинутый AI анализатор для профессионального анализа графиков
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.analysis_cache = {}
        
        # Настройки для больших датасетов
        max_rows = int(os.getenv('MAX_ROWS_ANALYSIS', '50000'))
        if len(df) > max_rows:
            self.df = df.sample(n=max_rows, random_state=42)
            print(f"⚠️ DataFrame sampled to {max_rows:,} rows for analysis")
    
    def analyze_chart(self, chart_data: Dict, chart_type: str, chart_title: str = "") -> str:
        """
        Анализирует график и возвращает профессиональный комментарий
        """
        try:
            # Получаем базовую информацию о данных
            data_summary = self._get_data_summary()
            
            # Генерируем анализ в зависимости от типа графика
            if chart_type == "histogram":
                return self._analyze_histogram(chart_data, chart_title, data_summary)
            elif chart_type == "scatter":
                return self._analyze_scatter(chart_data, chart_title, data_summary)
            elif chart_type == "bar":
                return self._analyze_bar_chart(chart_data, chart_title, data_summary)
            elif chart_type == "heatmap":
                return self._analyze_heatmap(chart_data, chart_title, data_summary)
            elif chart_type == "line":
                return self._analyze_line_chart(chart_data, chart_title, data_summary)
            elif chart_type == "gauge":
                return self._analyze_gauge_chart(chart_data, chart_title, data_summary)
            elif chart_type == "box":
                return self._analyze_box_plot(chart_data, chart_title, data_summary)
            elif chart_type == "pie":
                return self._analyze_pie_chart(chart_data, chart_title, data_summary)
            else:
                return self._analyze_generic_chart(chart_data, chart_title, data_summary)
                
        except Exception as e:
            return f"📊 **Chart Analysis**: {chart_title}\n\nThis visualization provides insights into your data structure and patterns. The chart shows key relationships and distributions that can inform your data-driven decisions."
    
    def _get_data_summary(self) -> str:
        """Получает краткое описание датасета"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(
            include=["object", "string", "category", "bool"]
        ).columns.tolist()
        
        summary = f"Dataset: {len(self.df):,} records, {len(self.df.columns)} columns"
        if numeric_cols:
            summary += f"\nNumeric columns: {', '.join(numeric_cols[:5])}"
        if categorical_cols:
            summary += f"\nCategorical columns: {', '.join(categorical_cols[:5])}"
        
        return summary
    
    def _analyze_histogram(self, chart_data: Dict, title: str, data_summary: str) -> str:
        """Анализирует гистограмму как data scientist"""
        try:
            # Извлекаем данные из графика
            x_data = chart_data.get('x', [])
            if not x_data:
                return self._generic_histogram_analysis(title)
            
            # Статистический анализ
            mean_val = np.mean(x_data)
            std_val = np.std(x_data)
            median_val = np.median(x_data)
            skewness = self._calculate_skewness(x_data)
            
            # Определяем тип распределения
            distribution_type = self._identify_distribution(x_data)
            
            # Анализ выбросов
            outliers = self._detect_outliers(x_data)
            
            analysis = f"""📊 **Histogram Analysis: {title}**

**Distribution Characteristics:**
• **Mean**: {mean_val:.2f} | **Median**: {median_val:.2f} | **Std Dev**: {std_val:.2f}
• **Distribution Type**: {distribution_type}
• **Skewness**: {skewness:.2f} ({self._interpret_skewness(skewness)})

**Data Quality Insights:**
• **Outliers Detected**: {len(outliers)} ({len(outliers)/len(x_data)*100:.1f}% of data)
• **Data Range**: {min(x_data):.2f} to {max(x_data):.2f}

**Professional Interpretation:**
{self._get_distribution_insights(distribution_type, skewness, outliers)}

**Recommendations:**
{self._get_histogram_recommendations(distribution_type, outliers, len(x_data))}"""
            
            return analysis
            
        except Exception as e:
            return self._generic_histogram_analysis(title)
    
    def _analyze_scatter(self, chart_data: Dict, title: str, data_summary: str) -> str:
        """Анализирует scatter plot"""
        try:
            x_data = chart_data.get('x', [])
            y_data = chart_data.get('y', [])
            
            if not x_data or not y_data:
                return self._generic_scatter_analysis(title)
            
            # Корреляционный анализ
            correlation = np.corrcoef(x_data, y_data)[0, 1]
            r_squared = correlation ** 2
            
            # Статистический анализ
            x_mean, y_mean = np.mean(x_data), np.mean(y_data)
            x_std, y_std = np.std(x_data), np.std(y_data)
            
            # Определяем тип связи
            relationship_type = self._identify_relationship(correlation)
            
            analysis = f"""📈 **Scatter Plot Analysis: {title}**

**Correlation Analysis:**
• **Pearson Correlation**: {correlation:.3f} ({self._interpret_correlation(correlation)})
• **R-squared**: {r_squared:.3f} ({r_squared*100:.1f}% of variance explained)
• **Relationship Type**: {relationship_type}

**Statistical Summary:**
• **X-axis**: Mean={x_mean:.2f}, Std={x_std:.2f}
• **Y-axis**: Mean={y_mean:.2f}, Std={y_std:.2f}
• **Data Points**: {len(x_data):,}

**Professional Insights:**
{self._get_scatter_insights(correlation, r_squared, len(x_data))}

**Business Implications:**
{self._get_scatter_business_insights(correlation, relationship_type)}"""
            
            return analysis
            
        except Exception as e:
            return self._generic_scatter_analysis(title)
    
    def _analyze_bar_chart(self, chart_data: Dict, title: str, data_summary: str) -> str:
        """Анализирует столбчатую диаграмму"""
        try:
            x_data = chart_data.get('x', [])
            y_data = chart_data.get('y', [])
            
            if not x_data or not y_data:
                return self._generic_bar_analysis(title)
            
            # Анализ категорий
            total_value = sum(y_data)
            max_value = max(y_data)
            min_value = min(y_data)
            max_category = x_data[y_data.index(max_value)]
            min_category = x_data[y_data.index(min_value)]
            
            # Концентрация данных
            top_3_sum = sum(sorted(y_data, reverse=True)[:3])
            concentration = top_3_sum / total_value * 100
            
            analysis = f"""📊 **Bar Chart Analysis: {title}**

**Category Performance:**
• **Total Value**: {total_value:,.0f}
• **Highest**: {max_category} ({max_value:,.0f})
• **Lowest**: {min_category} ({min_value:,.0f})
• **Range**: {max_value - min_value:,.0f}

**Data Concentration:**
• **Top 3 Categories**: {concentration:.1f}% of total value
• **Categories Count**: {len(x_data)}
• **Average per Category**: {total_value/len(x_data):,.0f}

**Professional Insights:**
{self._get_bar_insights(concentration, max_value, min_value, len(x_data))}

**Strategic Recommendations:**
{self._get_bar_recommendations(concentration, max_category, min_category)}"""
            
            return analysis
            
        except Exception as e:
            return self._generic_bar_analysis(title)
    
    def _analyze_heatmap(self, chart_data: Dict, title: str, data_summary: str) -> str:
        """Анализирует тепловую карту корреляций"""
        try:
            z_data = chart_data.get('z', [])
            if not z_data:
                return self._generic_heatmap_analysis(title)
            
            # Анализ корреляций
            strong_correlations = []
            moderate_correlations = []
            
            for i, row in enumerate(z_data):
                for j, val in enumerate(row):
                    if i != j:  # Исключаем диагональ
                        if abs(val) > 0.7:
                            strong_correlations.append(abs(val))
                        elif abs(val) > 0.3:
                            moderate_correlations.append(abs(val))
            
            analysis = f"""🔥 **Correlation Heatmap Analysis: {title}**

**Correlation Strength:**
• **Strong Correlations** (|r| > 0.7): {len(strong_correlations)}
• **Moderate Correlations** (0.3 < |r| < 0.7): {len(moderate_correlations)}
• **Variables Analyzed**: {len(z_data)}

**Key Insights:**
{self._get_heatmap_insights(strong_correlations, moderate_correlations)}

**Data Science Implications:**
{self._get_heatmap_implications(strong_correlations, moderate_correlations)}

**Recommendations:**
{self._get_heatmap_recommendations(strong_correlations, moderate_correlations)}"""
            
            return analysis
            
        except Exception as e:
            return self._generic_heatmap_analysis(title)
    
    def _analyze_gauge_chart(self, chart_data: Dict, title: str, data_summary: str) -> str:
        """Анализирует gauge chart (как на скриншоте)"""
        try:
            value = chart_data.get('value', 0)
            max_value = chart_data.get('max_value', 100)
            threshold = chart_data.get('threshold', 80)
            
            percentage = (value / max_value) * 100
            
            # Определяем статус
            if percentage >= threshold:
                status = "✅ Excellent"
                color = "green"
            elif percentage >= threshold * 0.7:
                status = "⚠️ Good"
                color = "yellow"
            else:
                status = "❌ Needs Attention"
                color = "red"
            
            analysis = f"""🎯 **Gauge Chart Analysis: {title}**

**Performance Metrics:**
• **Current Value**: {value:,.0f} / {max_value:,.0f}
• **Percentage**: {percentage:.1f}%
• **Status**: {status}
• **Threshold**: {threshold}%

**Professional Assessment:**
{self._get_gauge_assessment(percentage, threshold)}

**Business Impact:**
{self._get_gauge_business_impact(percentage, threshold)}

**Action Items:**
{self._get_gauge_actions(percentage, threshold)}"""
            
            return analysis
            
        except Exception as e:
            return self._generic_gauge_analysis(title)
    
    # Вспомогательные методы для анализа
    def _calculate_skewness(self, data):
        """Вычисляет асимметрию"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _identify_distribution(self, data):
        """Определяет тип распределения"""
        skewness = self._calculate_skewness(data)
        if abs(skewness) < 0.5:
            return "Normal-like distribution"
        elif skewness > 0.5:
            return "Right-skewed distribution"
        else:
            return "Left-skewed distribution"
    
    def _interpret_skewness(self, skewness):
        """Интерпретирует асимметрию"""
        if abs(skewness) < 0.5:
            return "approximately symmetric"
        elif skewness > 0.5:
            return "right-tailed"
        else:
            return "left-tailed"
    
    def _detect_outliers(self, data):
        """Обнаруживает выбросы"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return [x for x in data if x < lower_bound or x > upper_bound]
    
    def _identify_relationship(self, correlation):
        """Определяет тип связи"""
        if abs(correlation) > 0.7:
            return "Strong linear relationship"
        elif abs(correlation) > 0.3:
            return "Moderate linear relationship"
        else:
            return "Weak or no linear relationship"
    
    def _interpret_correlation(self, correlation):
        """Интерпретирует корреляцию"""
        if abs(correlation) > 0.7:
            strength = "strong"
        elif abs(correlation) > 0.3:
            strength = "moderate"
        else:
            strength = "weak"
        
        direction = "positive" if correlation > 0 else "negative"
        return f"{strength} {direction}"
    
    # Методы для генерации инсайтов
    def _get_distribution_insights(self, dist_type, skewness, outliers):
        """Генерирует инсайты о распределении"""
        insights = []
        
        if "Normal" in dist_type:
            insights.append("• Data follows a normal distribution, indicating natural variation")
        elif "Right-skewed" in dist_type:
            insights.append("• Right-skewed distribution suggests most values are below the mean")
        else:
            insights.append("• Left-skewed distribution indicates most values are above the mean")
        
        if len(outliers) > len(self.df) * 0.05:
            insights.append("• Significant outliers detected - investigate data quality")
        else:
            insights.append("• Outlier levels are within acceptable range")
        
        return "\n".join(insights)
    
    def _get_histogram_recommendations(self, dist_type, outliers, data_size):
        """Рекомендации для гистограммы"""
        recommendations = []
        
        if "Right-skewed" in dist_type:
            recommendations.append("• Consider log transformation for better analysis")
        elif "Left-skewed" in dist_type:
            recommendations.append("• Data may benefit from square root transformation")
        
        if len(outliers) > data_size * 0.1:
            recommendations.append("• Investigate and potentially remove outliers")
        
        recommendations.append("• Use this distribution for statistical modeling")
        
        return "\n".join(recommendations)
    
    def _get_scatter_insights(self, correlation, r_squared, data_size):
        """Инсайты для scatter plot"""
        insights = []
        
        if abs(correlation) > 0.7:
            insights.append("• Strong linear relationship detected - excellent for prediction")
        elif abs(correlation) > 0.3:
            insights.append("• Moderate relationship - useful for trend analysis")
        else:
            insights.append("• Weak relationship - consider non-linear models")
        
        insights.append(f"• {r_squared*100:.1f}% of variance explained by linear relationship")
        
        if data_size > 1000:
            insights.append("• Large sample size provides reliable correlation estimate")
        
        return "\n".join(insights)
    
    def _get_scatter_business_insights(self, correlation, relationship_type):
        """Бизнес-инсайты для scatter plot"""
        insights = []
        
        if "Strong" in relationship_type:
            insights.append("• High predictive power for business decisions")
            insights.append("• Consider this relationship for forecasting models")
        else:
            insights.append("• Limited predictive value - explore other variables")
        
        return "\n".join(insights)
    
    def _get_bar_insights(self, concentration, max_val, min_val, categories):
        """Инсайты для bar chart"""
        insights = []
        
        if concentration > 80:
            insights.append("• High concentration in top categories - Pareto principle applies")
        elif concentration > 60:
            insights.append("• Moderate concentration - balanced distribution")
        else:
            insights.append("• Low concentration - relatively even distribution")
        
        ratio = max_val / min_val if min_val > 0 else float('inf')
        if ratio > 10:
            insights.append("• High variance between categories - focus on top performers")
        
        return "\n".join(insights)
    
    def _get_bar_recommendations(self, concentration, max_cat, min_cat):
        """Рекомендации для bar chart"""
        recommendations = []
        
        if concentration > 70:
            recommendations.append(f"• Focus resources on top category: {max_cat}")
            recommendations.append("• Consider 80/20 analysis for optimization")
        else:
            recommendations.append("• Balanced approach across all categories")
        
        recommendations.append(f"• Investigate why {min_cat} underperforms")
        
        return "\n".join(recommendations)
    
    def _get_heatmap_insights(self, strong_corr, moderate_corr):
        """Инсайты для heatmap"""
        insights = []
        
        if len(strong_corr) > 5:
            insights.append("• Multiple strong correlations detected - multicollinearity risk")
        elif len(strong_corr) > 0:
            insights.append("• Some strong correlations present - good for prediction")
        else:
            insights.append("• No strong correlations - variables are relatively independent")
        
        if len(moderate_corr) > 10:
            insights.append("• Many moderate correlations - complex relationships exist")
        
        return "\n".join(insights)
    
    def _get_heatmap_implications(self, strong_corr, moderate_corr):
        """Импликации для heatmap"""
        implications = []
        
        if len(strong_corr) > 3:
            implications.append("• Consider dimensionality reduction techniques")
            implications.append("• Be cautious of overfitting in models")
        else:
            implications.append("• Good feature independence for modeling")
        
        return "\n".join(implications)
    
    def _get_heatmap_recommendations(self, strong_corr, moderate_corr):
        """Рекомендации для heatmap"""
        recommendations = []
        
        if len(strong_corr) > 0:
            recommendations.append("• Use correlation analysis for feature selection")
            recommendations.append("• Consider principal component analysis")
        
        recommendations.append("• Focus on variables with strong business relevance")
        
        return "\n".join(recommendations)
    
    def _get_gauge_assessment(self, percentage, threshold):
        """Оценка для gauge chart"""
        if percentage >= threshold:
            return "• Performance exceeds expectations - excellent results"
        elif percentage >= threshold * 0.8:
            return "• Performance is good but has room for improvement"
        else:
            return "• Performance below target - immediate action required"
    
    def _get_gauge_business_impact(self, percentage, threshold):
        """Бизнес-импакт для gauge chart"""
        if percentage >= threshold:
            return "• Positive impact on business objectives"
        else:
            return "• Potential negative impact - review strategy"
    
    def _get_gauge_actions(self, percentage, threshold):
        """Действия для gauge chart"""
        if percentage >= threshold:
            return "• Maintain current performance level"
            return "• Consider raising targets for next period"
        else:
            return "• Implement improvement initiatives"
            return "• Review and adjust strategy"
    
    # Generic analysis methods
    def _generic_histogram_analysis(self, title):
        return f"""📊 **Histogram Analysis: {title}**

This histogram shows the distribution of your data values. The shape of the distribution reveals important patterns about your dataset's characteristics and can guide your analytical approach.

**Key Points to Consider:**
• Distribution shape indicates data normality
• Outliers may represent important anomalies
• Skewness affects statistical modeling choices

**Professional Insight:** Use this distribution information to select appropriate statistical tests and modeling techniques."""

    def _generic_scatter_analysis(self, title):
        return f"""📈 **Scatter Plot Analysis: {title}**

This scatter plot reveals the relationship between two variables in your dataset. The pattern of points indicates correlation strength and direction.

**Key Points to Consider:**
• Correlation strength affects predictive power
• Linear vs non-linear relationships
• Outliers may indicate data quality issues

**Professional Insight:** Strong correlations suggest good predictive variables, while weak correlations may require feature engineering."""

    def _generic_bar_analysis(self, title):
        return f"""📊 **Bar Chart Analysis: {title}**

This bar chart compares values across different categories, making it easy to identify top performers and areas needing attention.

**Key Points to Consider:**
• Category performance ranking
• Value distribution across categories
• Concentration of results

**Professional Insight:** Use this for identifying focus areas and resource allocation decisions."""

    def _generic_heatmap_analysis(self, title):
        return f"""🔥 **Correlation Heatmap Analysis: {title}**

This heatmap visualizes correlations between multiple variables, helping identify relationships and potential multicollinearity.

**Key Points to Consider:**
• Strong correlations (red/blue) indicate related variables
• Weak correlations suggest independent variables
• Diagonal shows perfect self-correlation

**Professional Insight:** Use this for feature selection and understanding variable relationships in your dataset."""

    def _generic_gauge_analysis(self, title):
        return f"""🎯 **Gauge Chart Analysis: {title}**

This gauge chart provides a clear visual representation of a key performance metric against a target or threshold.

**Key Points to Consider:**
• Current performance level
• Distance from target
• Performance status (good/needs improvement)

**Professional Insight:** Gauge charts are excellent for executive dashboards and KPI monitoring."""

    def _analyze_line_chart(self, chart_data: Dict, title: str, data_summary: str) -> str:
        """Анализирует линейный график"""
        return f"""📈 **Line Chart Analysis: {title}**

This line chart shows trends over time or sequence, revealing patterns, growth, or decline in your data.

**Key Points to Consider:**
• Trend direction (increasing/decreasing/stable)
• Seasonal patterns or cycles
• Rate of change over time

**Professional Insight:** Use this for time series analysis and forecasting future trends."""

    def _analyze_box_plot(self, chart_data: Dict, title: str, data_summary: str) -> str:
        """Анализирует box plot"""
        return f"""📦 **Box Plot Analysis: {title}**

This box plot shows the distribution of data through quartiles, highlighting median, outliers, and data spread.

**Key Points to Consider:**
• Median and quartile positions
• Outlier detection
• Data spread and symmetry

**Professional Insight:** Excellent for comparing distributions across groups and detecting outliers."""

    def _analyze_pie_chart(self, chart_data: Dict, title: str, data_summary: str) -> str:
        """Анализирует круговую диаграмму"""
        return f"""🥧 **Pie Chart Analysis: {title}**

This pie chart shows the proportional composition of your data, making it easy to see relative sizes of different categories.

**Key Points to Consider:**
• Proportional distribution
• Dominant categories
• Category balance

**Professional Insight:** Best for showing parts of a whole, but limit to 5-7 categories for clarity."""







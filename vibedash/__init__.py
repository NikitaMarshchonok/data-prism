try:
    from flask import Blueprint
    
    # Создаем Blueprint для VibeDash
    vibedash_bp = Blueprint('vibedash', __name__, 
                           url_prefix='/vibedash',
                           template_folder='templates',
                           static_folder='static')
    
    # Импортируем routes после создания blueprint
    from . import routes
except ImportError:
    # Flask не установлен, создаем заглушку
    vibedash_bp = None

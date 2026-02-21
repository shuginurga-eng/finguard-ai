import google.generativeai as genai
genai.configure(api_key="AIzaSyDZ8WAIBqPcb7l3yuWEJxVeCNmxTY1YRlE")

print("--- ПРОВЕРКА ДОСТУПНЫХ МОДЕЛЕЙ ---")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Доступная модель: {m.name}")
except Exception as e:
    print(f"Ошибка: {e}")
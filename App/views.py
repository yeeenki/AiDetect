from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login, logout, authenticate
import joblib
from .ml.text_detect import predict_text
import json
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from .models import DetectionResult
from django.contrib.auth.decorators import login_required
import os
import sys
from django.views.decorators.csrf import csrf_exempt
from transformers import pipeline as hf_pipeline
from PIL import Image
import io
from .models import DetectionResult, MediaDetectionResult



model = joblib.load("App/ml/model.pkl")
tfidf = joblib.load("App/ml/tfidf.pkl")
feat_cols = joblib.load("App/ml/feat_cols.pkl")

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'users/register.html', {'form': form})

def user_login(request):
    error = None 
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            error = 'Неверный логин или пароль'  
    return render(request, 'users/login.html', {'error': error})

def user_logout(request):
    logout(request)
    return redirect('home')

def user_home(request):
    return render(request, 'users/home.html')


def detect_ai(request):
    result = None

    if request.method == "POST":
        text = request.POST.get("text")

        if text:
            result = predict_text(text, model, tfidf, feat_cols)
            result["final_probability"] *= 100  # важно!

    return render(request, "detect.html", {"result": result})

@login_required
@require_POST
def detect_api(request):
    try:
        data = json.loads(request.body)
        text = data.get("text", "").strip()

        if not text:
            return JsonResponse({"error": "Текст пустой."}, status=400)

        result = predict_text(text, model, tfidf, feat_cols)

        # сохраняем результат в PostgreSQL
        DetectionResult.objects.create(
            user=request.user,
            text=text,
            verdict=result.get("verdict", ""),
            final_probability=result.get("final_probability", 0),
            statistical_score=result.get("statistical_score", 0),
            ml_probability=result.get("ml_probability", 0),
            details=result.get("details", {})
        )

        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Неверный JSON."}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def history_view(request):
    analyses = DetectionResult.objects.filter(user=request.user).order_by('-created_at')
    for item in analyses:
        item.ai_percent = round(item.final_probability * 100)
        item.human_percent = 100 - item.ai_percent

    media_analyses = MediaDetectionResult.objects.filter(user=request.user).order_by('-created_at')
    for item in media_analyses:
        item.confidence_percent = round(item.confidence * 100)

    return render(request, 'users/history.html', {
        'analyses': analyses,
        'media_analyses': media_analyses
    })


# Подключаем детектор
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml', 'audio_detect'))
from pipeline import AudioDetector

detector = AudioDetector(
    model_path=os.path.join(os.path.dirname(__file__), 'ml', 'audio_detect', 'detector.pkl')
)

def detector_audio_page(request):
    """Страница с формой загрузки аудио"""
    return render(request, 'detector.html')

@csrf_exempt
def detect_audio(request):
    """API endpoint — принимает аудио, возвращает результат"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Только POST'}, status=405)

    audio_file = request.FILES.get('file')
    if not audio_file:
        return JsonResponse({'error': 'Файл не загружен'}, status=400)

    # Сохраняем временно
    tmp_path = os.path.join(os.path.dirname(__file__), f'tmp_{audio_file.name}')
    with open(tmp_path, 'wb') as f:
        for chunk in audio_file.chunks():
            f.write(chunk)

    # Анализируем
    result = detector.predict(tmp_path)
    os.remove(tmp_path)

    if request.user.is_authenticated:
        MediaDetectionResult.objects.create(
            user=request.user,
            media_type='audio',
            filename=audio_file.name,
            label=result['label'],
            confidence=result['confidence']
        )

    return JsonResponse({
        'label': result['label'],           # "natural" или "ai_generated"
        'confidence': result['confidence'],  # например 0.98
    })
    

# ── Детекция фото ──────────────────────────────────────────────────────

# Загружаем модель один раз при старте
image_detector = hf_pipeline(
    "image-classification",
    model="umm-maybe/AI-image-detector"
)

@csrf_exempt
def detect_media_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Только POST'}, status=405)
    image_file = request.FILES.get('file')
    if not image_file:
        return JsonResponse({'error': 'Файл не загружен'}, status=400)
    try:
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        results = image_detector(image)

        ai_score = next((r['score'] for r in results if r['label'] == 'artificial'), 0)
        human_score = next((r['score'] for r in results if r['label'] == 'human'), 0)
        label = 'ai_generated' if ai_score > 0.5 else 'natural'
        confidence = round(ai_score if label == 'ai_generated' else human_score, 4)
        if request.user.is_authenticated:
            MediaDetectionResult.objects.create(
                user=request.user,
                media_type='image',
                filename=image_file.name,
                label=label,
                confidence=confidence
            )
        return JsonResponse({
            'label': label,
            'confidence': confidence,
            'ai_score': round(ai_score * 100, 1),
            'human_score': round(human_score * 100, 1),
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    

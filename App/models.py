from django.db import models
from django.contrib.auth.models import User

class DetectionResult(models.Model):
    user              = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    text              = models.TextField()
    verdict           = models.CharField(max_length=100)
    final_probability = models.FloatField()
    statistical_score = models.FloatField()
    ml_probability    = models.FloatField()
    details           = models.JSONField()
    created_at        = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user} — {self.verdict} ({self.final_probability:.0%})"
class MediaDetectionResult(models.Model):
    TYPE_CHOICES = [
        ('audio', 'Аудио'),
        ('image', 'Изображение'),
    ]

    user       = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    media_type = models.CharField(max_length=10, choices=TYPE_CHOICES)
    filename   = models.CharField(max_length=255)
    label      = models.CharField(max_length=50)   # "ai_generated" или "natural"
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user} — {self.media_type} — {self.label} ({self.confidence:.0%})"
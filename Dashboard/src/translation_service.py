import json
import os
import hashlib
import time

class TranslationService:
    """
    Core Backend Service cho việc quản lý dịch thuật (Pali/Hán -> Việt).
    Tích hợp cơ chế Local Caching (giảm thiểu API calls) và QC Validation.
    """
    def __init__(self, cache_file='final/translation_cache.json'):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self._qc_stats = {'cache_hits': 0, 'api_calls': 0, 'missing': 0}

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _generate_hash(self, text):
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _mock_translate_api(self, text, source_lang):
        """
        Sử dụng Google Translate API (Free endpoint) để dịch Pali/Hán sang tiếng Việt.
        """
        import requests
        try:
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                "client": "gtx",
                "sl": "auto", # Tự động phát hiện Pali/Hán
                "tl": "vi",   # Dịch sang tiếng Việt
                "dt": "t",
                "q": text
            }
            # Thêm timeout và headers để tránh bị chặn
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, params=params, headers=headers, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                # Google Translate trả về list các câu, cần nối lại
                translated_text = ''.join([sentence[0] for sentence in result[0]])
                
                # Thêm prefix nhỏ để QA dễ theo dõi nguồn dịch
                prefix = "[Google Dịch Pali]" if source_lang == 'pali' else "[Google Dịch Hán]"
                return f"{prefix} {translated_text}"
            else:
                return f"[Lỗi API {response.status_code}]"
        except Exception as e:
            return f"[Lỗi Kết Nối] {str(e)}"

    def translate_batch(self, texts, source_lang):
        """
        Dịch theo lô (batch) để tối ưu hiệu năng API.
        Được QC giám sát chặt chẽ tỷ lệ Cache Hit.
        """
        results = []
        new_translations = False
        
        for text in texts:
            if not isinstance(text, str) or text.strip() == "" or text == "[THIẾU/TRỐNG]":
                results.append("[THIẾU/TRỐNG]")
                continue

            text_hash = self._generate_hash(text)
            if text_hash in self.cache:
                self._qc_stats['cache_hits'] += 1
                results.append(self.cache[text_hash])
            else:
                self._qc_stats['api_calls'] += 1
                translated = self._mock_translate_api(text, source_lang)
                results.append(translated)
                
                # CHỈ LƯU CACHE NẾU KHÔNG LỖI
                if not translated.startswith("[Lỗi"):
                    self.cache[text_hash] = translated
                    new_translations = True

        if new_translations:
            self._save_cache()
            
        return results

    def print_qc_report(self):
        """QC Role: Xuất báo cáo kiểm định chất lượng dịch thuật"""
        print("\n" + "="*40)
        print("📊 [QC REPORT] BÁO CÁO VẬN HÀNH DỊCH THUẬT")
        print("="*40)
        total_requests = self._qc_stats['cache_hits'] + self._qc_stats['api_calls']
        print(f"Tổng số đoạn văn bản xử lý : {total_requests}")
        print(f"Tỷ lệ Cache Hit            : {self._qc_stats['cache_hits']}")
        print(f"Số lượng gọi API mới       : {self._qc_stats['api_calls']}")
        if total_requests > 0:
            savings = (self._qc_stats['cache_hits'] / total_requests) * 100
            print(f"Tối ưu chi phí API         : Tiết kiệm được {savings:.1f}%")
        print("="*40 + "\n")

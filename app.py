import os  # التعامل مع المسارات | Path handling
import re  # التعبيرات النمطية للتقطيع | Regular expressions for tokenization
import gradio as gr  # مكتبة بناء الواجهة | UI library
import torch  # مكتبة بايتورتش | PyTorch library
from transformers import AutoModelForTokenClassification, AutoTokenizer  # تحميل النموذج والمقطّع | Load model and tokenizer

MODEL_DIR = os.environ.get("NER_MODEL_ID", "MohamedBelal-AI/deberta-v3-base-conll2003-ner")  # مستودع الموديل على الهاب | Model repository on the Hub
MAX_LENGTH = 256  # أقصى طول للتوكنز الفرعية | Max sub-word length

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # تحديد الجهاز | Select device
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)  # تحميل المقطّع | Load the tokenizer
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, dtype=torch.float32).to(device)  # تحميل النموذج بدقة fp32 | Load the model in fp32
model.eval()  # وضع التقييم | Evaluation mode

ENTITY_COLORS = {"PER": "#ff6b6b", "ORG": "#4dabf7", "LOC": "#51cf66", "MISC": "#fcc419"}  # ألوان الكيانات | Entity colors
TOKEN_PATTERN = re.compile(r"(?:[A-Za-z]\.){2,}|\d+(?:[.,]\d+)+|\w+(?:-\w+)*|['’]s\b|[^\w\s]")  # تقطيع بأسلوب CoNLL | CoNLL-style tokenization


def group_entities(spans, labels):  # دالة تجميع الوسوم إلى كيانات كاملة | Group word labels into complete entities
    entities = []  # قائمة الكيانات | Entity list
    current = None  # الكيان قيد البناء | Entity being built
    for (start, end), label in zip(spans, labels):  # المرور على الكلمات بمواضعها | Iterate words with offsets
        if label == "O":  # لو مش كيان | If not an entity
            current = None  # قفل أي كيان مفتوح | Close any open entity
            continue  # الانتقال للكلمة التالية | Move to next word
        entity_type = label.split("-")[-1]  # نوع الكيان فقط | Entity type only
        if label.startswith("B-") or current is None or current["entity"] != entity_type:  # بداية كيان جديد | A new entity starts
            current = {"entity": entity_type, "start": start, "end": end}  # إنشاء كيان بمواضع حقيقية | New entity with real offsets
            entities.append(current)  # إضافته للقائمة | Add it to the list
        else:  # استمرار نفس الكيان | Same entity continues
            current["end"] = end  # توسيع نهاية الكيان | Extend the entity end
    return entities  # إرجاع الكيانات | Return the entities


def predict_entities(sentence):  # دالة التنبؤ الرئيسية | Main prediction function
    sentence = sentence or ""  # التعامل مع الإدخال الفارغ | Handle empty input
    matches = list(TOKEN_PATTERN.finditer(sentence))  # استخراج الكلمات مع مواضعها | Extract words with offsets
    if not matches:  # لو مفيش كلمات | If there are no words
        return {"text": sentence, "entities": []}  # إرجاع نتيجة فاضية | Return an empty result

    words = [m.group() for m in matches]  # نصوص الكلمات | Word strings
    spans = [(m.start(), m.end()) for m in matches]  # مواضع الكلمات في النص الأصلي | Word offsets in the original text
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)  # تقطيع للنموذج | Tokenize for the model
    word_ids = inputs.word_ids(batch_index=0)  # ربط التوكنز الفرعية بالكلمات | Map sub-words to words
    model_inputs = {"input_ids": inputs["input_ids"].to(device), "attention_mask": inputs["attention_mask"].to(device)}  # نقل المدخلات للجهاز | Move inputs to device

    with torch.no_grad():  # بدون جراديانت | No gradients
        logits = model(**model_inputs).logits  # حساب المخرجات | Compute logits
    predictions = torch.argmax(logits, dim=-1)[0].tolist()  # أعلى احتمال لكل توكن | Highest-probability class per token

    word_labels = ["O"] * len(words)  # وسم افتراضي لكل كلمة | Default label per word
    seen_words = set()  # الكلمات التي أخذت وسمها | Words already labeled
    for idx, word_id in enumerate(word_ids):  # المرور على التوكنز الفرعية | Iterate sub-words
        if word_id is not None and word_id not in seen_words:  # أول جزء من كل كلمة فقط | First sub-word per word only
            word_labels[word_id] = model.config.id2label[predictions[idx]]  # تخزين وسم الكلمة | Store the word label
            seen_words.add(word_id)  # تعليم الكلمة كمعالجة | Mark word as processed

    return {"text": sentence, "entities": group_entities(spans, word_labels)}  # الإخراج بصيغة HighlightedText | Output in HighlightedText format


demo = gr.Interface(  # بناء الواجهة | Build the interface
    fn=predict_entities,  # الدالة المنفذة | Function to run
    inputs=gr.Textbox(label="اكتب جملة | Enter a sentence", placeholder="Mohamed Belal works as an AI Engineer at Google in Dubai", lines=2),  # خانة الإدخال | Input box
    outputs=gr.HighlightedText(label="الكيانات المكتشفة | Detected Entities", color_map=ENTITY_COLORS),  # عرض الكيانات الملونة | Colored entity output
    title="🔍 Named Entity Recognition (DeBERTa-v3)",  # العنوان | Title
    description="أدخل جملة بالإنجليزية وشوف الكيانات (أشخاص، مؤسسات، أماكن، متفرقات) ملونة تلقائياً. | Enter an English sentence and see entities (people, organizations, locations, misc) auto-highlighted.",  # الوصف | Description
    examples=[  # أمثلة جاهزة | Ready-made examples
        ["Mohamed Belal works as an AI Engineer at Google in Dubai"],  # مثال 1 | Example 1
        ["Barack Obama visited Paris last week to meet officials from the United Nations"],  # مثال 2 | Example 2
        ["Angela Merkel met Emmanuel Macron in Berlin, Paris, and Rome."],  # مثال بترقيم | Example with punctuation
    ],  # نهاية الأمثلة | End of examples
)  # نهاية الواجهة | End of interface

if __name__ == "__main__":  # نقطة التشغيل | Entry point
    demo.launch()  # تشغيل التطبيق | Launch the app

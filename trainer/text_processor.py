"""文本编码模块：支持拼音和字符两种模式"""
import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# ============================================================
# 拼音模式符号表
# ============================================================
INITIALS = [
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x",
    "zh", "ch", "sh", "r", "z", "c", "s",
    "y", "w",
]

FINALS = [
    "a", "o", "e", "i", "u", "v",
    "ai", "ei", "ao", "ou",
    "an", "en", "ang", "eng", "ong",
    "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong",
    "ua", "uo", "uai", "ui", "uan", "un", "uang",
    "ve", "van", "vn",
    "er",
]

# 声调标记（0=轻声, 1-4=四声）
TONES = ["_0", "_1", "_2", "_3", "_4"]

# 标点符号
PUNCTUATIONS = [",", ".", "!", "?", ";", ":", "'", "\"", "(", ")", "-", "...", "，", "。", "！", "？", "；", "：", "、", "（", "）", "—", "…", "'", "'", """, """]

# 特殊 token
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sil>", "<sp>", "<unk>"]

# 构建完整符号表（拼音模式）
PINYIN_SYMBOLS = SPECIAL_TOKENS + INITIALS + FINALS + TONES + PUNCTUATIONS
PINYIN_SYMBOL_TO_ID = {s: i for i, s in enumerate(PINYIN_SYMBOLS)}
PINYIN_VOCAB_SIZE = len(PINYIN_SYMBOLS)

# ============================================================
# 字符模式符号表
# ============================================================
# 常用汉字（GB2312 一级 + 二级，约 3755 个）
# 这里生成常用汉字范围，实际使用时按需扩展
COMMON_CHARS = list(range(0x4e00, 0x9fff))  # CJK 统一汉字基本区
CHAR_SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
CHAR_SYMBOL_TO_ID = {s: i for i, s in enumerate(CHAR_SPECIAL)}
# 常用字符从 ID 4 开始
for cp in COMMON_CHARS:
    CHAR_SYMBOL_TO_ID[chr(cp)] = len(CHAR_SYMBOL_TO_ID)
# 添加标点
for p in PUNCTUATIONS:
    if p not in CHAR_SYMBOL_TO_ID:
        CHAR_SYMBOL_TO_ID[p] = len(CHAR_SYMBOL_TO_ID)
# 添加 ASCII 可打印字符
for c in range(0x20, 0x7f):
    ch = chr(c)
    if ch not in CHAR_SYMBOL_TO_ID:
        CHAR_SYMBOL_TO_ID[ch] = len(CHAR_SYMBOL_TO_ID)

CHAR_VOCAB_SIZE = len(CHAR_SYMBOL_TO_ID)

# ID 到符号的反向映射
PINYIN_ID_TO_SYMBOL = {i: s for s, i in PINYIN_SYMBOL_TO_ID.items()}
CHAR_ID_TO_SYMBOL = {i: s for s, i in CHAR_SYMBOL_TO_ID.items()}

# 拼音声母韵母映射（pypinyin 输出 → 我们的符号表）
# 处理 pypinyin 的特殊拼音格式
_PINYIN_REPLACE = {
    "lv": "v", "nv": "v", "lve": "ve", "nve": "ve",
}


def get_symbol_count(mode: str = "pinyin") -> int:
    """获取符号表大小。"""
    if mode == "pinyin":
        return PINYIN_VOCAB_SIZE
    return CHAR_VOCAB_SIZE


def _split_pinyin(syllable: str) -> tuple:
    """将一个拼音音节拆分为声母 + 韵母。"""
    s = syllable.lower().strip()
    if not s:
        return "", ""

    # 尝试匹配声母
    # 按长度从长到短匹配（zh, ch, sh 优先）
    for initial in sorted(INITIALS, key=len, reverse=True):
        if s.startswith(initial):
            final = s[len(initial):]
            if final in [f for f in FINALS]:
                return initial, final
            # 尝试去掉声调后匹配
            if final and final[-1].isdigit():
                tone_char = final[-1]
                final_body = final[:-1]
                if final_body in [f for f in FINALS]:
                    return initial, final  # 保留声调数字
            # 没有匹配到韵母，但声母有效
            return initial, final

    # 没有声母（零声母）
    return "", s


def _pinyin_to_symbols(text: str) -> List[str]:
    """将中文文本转换为拼音符号序列。"""
    try:
        from pypinyin import pinyin, Style
    except ImportError:
        logger.warning("pypinyin 未安装，回退到字符模式")
        return _char_to_symbols(text)

    # 清理文本
    text = text.strip()
    if not text:
        return ["<bos>", "<eos>"]

    symbols = ["<bos>"]

    # 分句处理，保留标点
    # 使用 pypinyin 获取带声调拼音
    py_result = pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)

    for item in py_result:
        syllable = item[0]
        if not syllable:
            continue

        # 检查是否为标点
        if syllable in PUNCTUATIONS:
            symbols.append(syllable)
            continue

        # 检查是否为空格
        if syllable.strip() == "":
            symbols.append("<sp>")
            continue

        # 提取声调数字
        tone = "0"
        body = syllable
        if syllable and syllable[-1].isdigit():
            tone = syllable[-1]
            body = syllable[:-1]

        # 特殊替换
        body = _PINYIN_REPLACE.get(body, body)

        # 拆分声母韵母
        initial, final = _split_pinyin(body)
        if initial and initial in PINYIN_SYMBOL_TO_ID:
            symbols.append(initial)
        if final:
            # 韵母可能需要清理
            final_clean = final.lower().strip()
            if final_clean in PINYIN_SYMBOL_TO_ID:
                symbols.append(final_clean)
            elif final_clean:
                # 未在符号表中，尝试逐字符
                symbols.append("<unk>")

        # 声调
        tone_mark = f"_{tone}"
        if tone_mark in PINYIN_SYMBOL_TO_ID:
            symbols.append(tone_mark)
        else:
            symbols.append("_0")

    symbols.append("<eos>")
    return symbols


def _char_to_symbols(text: str) -> List[str]:
    """将文本转换为字符 ID 序列。"""
    text = text.strip()
    if not text:
        return ["<bos>", "<eos>"]

    symbols = ["<bos>"]
    for ch in text:
        if ch in CHAR_SYMBOL_TO_ID:
            symbols.append(ch)
        elif ch == " ":
            symbols.append("<sp>")
        else:
            symbols.append("<unk>")
    symbols.append("<eos>")
    return symbols


def text_to_sequence(text: str, mode: str = "pinyin"):
    """将文本转换为整数序列张量。

    Args:
        text: 输入文本
        mode: "pinyin" 或 "char"

    Returns:
        list: 符号 ID 列表
    """
    if mode == "pinyin":
        symbols = _pinyin_to_symbols(text)
        return [PINYIN_SYMBOL_TO_ID.get(s, PINYIN_SYMBOL_TO_ID["<unk>"]) for s in symbols]
    else:
        symbols = _char_to_symbols(text)
        return [CHAR_SYMBOL_TO_ID.get(s, CHAR_SYMBOL_TO_ID["<unk>"]) for s in symbols]


def sequence_to_text(seq: list, mode: str = "pinyin") -> str:
    """将序列转回文本（调试用）。"""
    if mode == "pinyin":
        return " ".join(PINYIN_ID_TO_SYMBOL.get(i, "?") for i in seq)
    else:
        id_to = CHAR_ID_TO_SYMBOL
        return "".join(id_to.get(i, "?") for i in seq if i > 0)


# ============================================================
# 预估 duration（基于文本长度的简单估计）
# ============================================================
def estimate_duration_from_text(text: str, mode: str = "pinyin") -> List[int]:
    """为每个音素/字符估计一个粗略的 duration（帧数）。

    简化方案：每个符号分配固定帧数，标点分配更多。

    Returns:
        list: 每个符号对应的帧数估计
    """
    if mode == "pinyin":
        symbols = _pinyin_to_symbols(text)
        symbol_to_id = PINYIN_SYMBOL_TO_ID
    else:
        symbols = _char_to_symbols(text)
        symbol_to_id = CHAR_SYMBOL_TO_ID

    durations = []
    for s in symbols:
        sid = symbol_to_id.get(s, 0)
        if s in ("<pad>",):
            durations.append(0)
        elif s in PUNCTUATIONS:
            durations.append(8)  # 标点停顿
        elif s in ("<sil>", "<sp>"):
            durations.append(6)
        elif s in ("<bos>", "<eos>"):
            durations.append(1)
        else:
            durations.append(4)  # 普通音素

    return durations

import unittest
from silverspeak.homoglyphs.random_attack import random_attack
from silverspeak.homoglyphs.normalize import normalize_text as normalize

class TestRandomAttackIntegration(unittest.TestCase):

    def test_random_attack_and_normalize(self):
        original_text = "This is a test sentence with some unique characters: ä, ö, ü."

        # Apply random attack
        attacked_text = random_attack(original_text, percentage=0.2, random_seed=42)

        # Normalize the attacked text
        normalized_text = normalize(attacked_text)

        # Assert that the normalized text matches the original
        self.assertEqual(normalized_text, original_text)

    def test_random_attack_and_normalize_empty_string(self):
        original_text = ""

        # Apply random attack
        attacked_text = random_attack(original_text, percentage=0.2, random_seed=42)

        # Normalize the attacked text
        normalized_text = normalize(attacked_text)

        # Assert that the normalized text matches the original
        self.assertEqual(normalized_text, original_text)

    def test_random_attack_and_normalize_special_characters(self):
        original_text = "!@#$%^&*()_+-=[]{}|;':,./<>?"

        # Apply random attack
        attacked_text = random_attack(original_text, percentage=0.2, random_seed=42)

        # Normalize the attacked text
        normalized_text = normalize(attacked_text)

        # Assert that the normalized text matches the original
        self.assertEqual(normalized_text, original_text)

    def test_random_attack_and_normalize_single_script(self):
        multilingual_phrases = [
            "The quick brown fox jumps over the lazy dog while the sun sets behind the mountains, painting the sky in hues of orange and purple.",  # English
            "Dans un petit village niché au cœur des montagnes, les habitants se réunissent chaque soir pour partager des histoires autour d'un feu de camp.",  # French
            "In einem kleinen Dorf, das tief im Wald versteckt liegt, erzählen sich die Menschen Legenden von längst vergangenen Zeiten.",  # German
            "En una tarde tranquila de verano, los niños juegan en el parque mientras los adultos disfrutan de una conversación bajo la sombra de los árboles.",  # Spanish
            "夕暮れ時、静かな村では子供たちが遊び、大人たちは木陰でお茶を飲みながら談笑しています。",  # Japanese
            "在一个宁静的夏日黄昏，孩子们在公园里玩耍，大人们在树荫下聊天，享受着微风的轻拂。",  # Chinese
            "В тихий летний вечер дети играют на улице, а взрослые сидят на скамейках, обсуждая последние новости.",  # Russian
            "조용한 여름 저녁, 아이들은 공원에서 놀고 어른들은 나무 그늘 아래에서 대화를 나누며 시간을 보냅니다.",  # Korean
            "एक शांत गर्मी की शाम में, बच्चे पार्क में खेलते हैं और वयस्क पेड़ों की छाया में बैठकर बातचीत का आनंद लेते हैं।",  # Hindi
            "في مساء صيفي هادئ، يلعب الأطفال في الحديقة بينما يجلس الكبار تحت ظلال الأشجار يتبادلون الأحاديث.",  # Arabic
            "Numa tarde de verão tranquila, as crianças brincam no parque enquanto os adultos conversam à sombra das árvores.",  # Portuguese
            "Σε ένα ήσυχο καλοκαιρινό απόγευμα, τα παιδιά παίζουν στο πάρκο ενώ οι ενήλικες συζητούν κάτω από τη σκιά των δέντρων.",  # Greek
            "Sıcak bir yaz akşamında, çocuklar parkta oynarken yetişkinler ağaçların gölgesinde sohbet ediyor.",  # Turkish
            "U mirnoj ljetnoj večeri, djeca se igraju u parku dok odrasli razgovaraju u hladu drveća.",  # Serbian
            "Dans un petit village, les enfants jouent dans les champs tandis que les adultes préparent un grand festin pour célébrer la récolte.",  # French (alternative)
        ]

        for phrase in multilingual_phrases:
            with self.subTest(msg=f"Testing phrase: {phrase}"):
                # Apply random attack
                attacked_text = random_attack(phrase, percentage=0.2, random_seed=42)

                # Normalize the attacked text
                normalized_text = normalize(attacked_text)

                # Assert that the normalized text matches the original
                self.assertEqual(normalized_text, phrase)

    def test_random_attack_and_normalize_mixed_scripts(self):
        mixed_script_phrases = [
            "En una tarde tranquila, los niños juegan en el parque mientras los adultos disfrutan de una conversación bajo la sombra de los árboles, y uno de ellos menciona: \"这是一个测试句子。\"",  # Spanish with Chinese
            "On a peaceful summer evening, children play in the park while adults chat under the trees, and someone says: \"Это тестовое предложение.\"",  # English with Russian
            "Par une soirée d'été paisible, les enfants jouent dans le parc tandis que les adultes discutent sous les arbres, et l'un d'eux ajoute : \"これはテスト文です。\"",  # French with Japanese
            "An einem ruhigen Sommerabend spielen Kinder im Park, während Erwachsene unter den Bäumen plaudern, und jemand bemerkt: \"هذا جملة اختبارية.\"",  # German with Arabic
            "Numa tarde de verão tranquila, as crianças brincam no parque enquanto os adultos conversam à sombra das árvores, e alguém comenta: \"Αυτή είναι μια δοκιμαστική πρόταση.\"",  # Portuguese with Greek
        ]

        for phrase in mixed_script_phrases:
            with self.subTest(msg=f"Testing mixed script phrase: {phrase}"):
                # Apply random attack
                attacked_text = random_attack(phrase, percentage=0.2, random_seed=42)

                # Normalize the attacked text
                normalized_text = normalize(attacked_text)

                # Assert that the normalized text matches the original
                self.assertEqual(normalized_text, phrase)

if __name__ == "__main__":
    unittest.main()
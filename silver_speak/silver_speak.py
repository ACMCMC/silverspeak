# %%
import random

SPACES_MAP = [
    "\u2000",
    "\u2002",
    "\u2005\u200A\u2006",
    "\u2006\u2006\u2006",
    "\u2007",
    "\u202f\u2006\u200A",
    "\u205f\u2006\u2006",
]
#SPACES_MAP = [
#    "\u2007\u2062",
#]

def replace_spaces(text):
    # Replaces all spaces in text with a random space from the SPACES_MAP
    perturbed_text = ""
    words = text.split(" ")
    for word in words:
        perturbed_text += word + random.choice(SPACES_MAP)
    return perturbed_text[:-1]  # Remove last space


def convert_to_char_from_hex(hex_num):
    # append 0x to hex num string to convert it.
    hex_num = "0x" + hex_num.strip(" ")
    # get the actual integer of the specific hex num with base 16.
    hex_num = int(hex_num, base=16)
    # finally get the actual character stored for specific hex char representation.
    hex_num = chr(hex_num)
    return hex_num


def read_map():
    """This function reads the identical_map.txt file and returns the final_map list."""
    with open("./identical_map.txt") as f:
        array = []
        lines = f.readlines()
        for line in lines:
            line = line.split(":")[1]
            line = line.strip("\n")
            line = line.strip(" ")
            array.append(line)
    final_map = []
    for record in array:
        line = record.split(",")
        final_map.append(line)
    return final_map


def replace_characters_by_equivalents(final_map, text):
    """This is an attack where we replace only the negative sentiment words found in negative-words list."""
    
    # Replace all chars in text with a random char from the final_map
    rewritten_text = ''
    rewrite = True
    for word in text.split(" "):
        if random.random() < 0.0:
            if random.random() < 0.4 and not rewrite:
                rewrite = not rewrite # flip the rewrite flag
            else:
                rewrite = not rewrite # flip the rewrite flag
        if not rewrite:
            rewritten_text += word + " "
            continue
        for char in word:
            if char >= "A" and char <= "Z":
                # find the sublist pointer from the final_map list.
                pointer = ord(char) - ord("A")
                # get random char from the final_map to replace the original char.
                random_choice = random.randrange(0, len(final_map[pointer]))

                # get chosen hex num
                chosen_char = str(final_map[pointer][random_choice])

                chosen_char = convert_to_char_from_hex(chosen_char)

                # write modified char to perturbed file.
                rewritten_text += chosen_char

            elif char >= "a" and char <= "z":
                # find the sublist pointer from the final_map list.
                pointer = ord(char) - ord("a") + 26
                # get random char from the final_map to replace the original char.
                random_choice = random.randrange(0, len(final_map[pointer]))

                # get chosen hex num
                chosen_char = str(final_map[pointer][random_choice])

                chosen_char = convert_to_char_from_hex(chosen_char)

                # write modified char to perturbed file.
                rewritten_text += chosen_char

            else:
                # other type of character so write it to file as it is.
                rewritten_text += char
        rewritten_text += " "

    return rewritten_text

final_map = read_map()
def rewrite_attack(text, do_replace_chars=True, do_replace_spaces=True):
    rewritten_text = text
    if do_replace_chars:
        rewritten_text = replace_characters_by_equivalents(final_map, rewritten_text)
    if do_replace_spaces:
        rewritten_text = replace_spaces(rewritten_text)
    return rewritten_text

if __name__ == "__main__":
    # legitimate_text = input("Give the legitimate/original text to be perturbed in 1 line:\n")
    original_text = """The following is a transcript from The Guardian's interview with the British ambassador to the UN, John Baird. Baird: The situation in Syria is very dire. We have a number of reports of chemical weapons being used in the country. The Syrian opposition has expressed their willingness to use chemical weapons. We have a number of people who have been killed, many of them civilians. I think it is important to understand this. There are many who are saying that the chemical weapons used in Syria are not only used to destroy people but also to destroy the Syrian people. The Syrian people have been suffering for many years. The regime is responsible for that suffering. They have been using chemical weapons. They have killed many people, and they continue to kill many more. I think that the international community has to take a position that the Assad regime has a responsibility for that suffering. It must take a stand that we are not going to allow the Syrian government to use chemical weapons on civilians, that we are not going to allow them, and that we do not condone their use. We have a lot of people who believe that the regime is responsible for this suffering, and that they are responsible for this suffering, and that they are responsible for the use of chemical weapons. I think that we need to be clear about that. We must be clear that the use of chemical weapons by any country, including Russia and Iran, is a violation of international law. We are not going to tolerate that. We do not tolerate that. And we have the responsibility to ensure that the world doesn't allow the Assad regime to use chemical weapons against civilians. Baird: It seems that there are a range of people that are saying that we are not allowed to use chemical weapons in Syria. There are many who say we are not allowed to use chemical weapons in Syria. I think there are a lot of people that are saying that we are not allowed to use chemical weapons in Syria. I think that we have to take a stand that we are not going to allow the Assad regime to use chemical weapons on civilians, that we are not going to tolerate that. We have to take a stand that we are not going to allow Russia and Iran to use chemical weapons on civilians. Baird: I think it is important for us to understand that the use of chemical weapons in Syria is an extremely dangerous situation. I think there has been very little information from the UN that the regime has used any chemical weapons. We have not seen any evidence that they are using them. We have to understand that the use of chemical weapons is very dangerous."""

    rewritten_text = rewrite_attack(original_text, replace_chars=True, replace_spaces=False)
    print("\n========================\n")
    print(rewritten_text)
    print("\n========================\n")

    # Tokenize the text using GPT2Tokenizer and print its decoded form
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('t5-base')
    print(tok.decode(tok.encode(rewritten_text)))

# %%
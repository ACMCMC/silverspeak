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
from silver_speak.identical_map import chars_map
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

from silver_speak.utils import encode_text, loglikelihood, replace_characters, decode_tokens
def decrease_loglikelihood_replace_characters_by_equivalents(chars_map, text, patience=10):
    encoded_text = encode_text(text)
    loglikelihoods = loglikelihood(encoded_text)
    print(f'Mean starting loglikelihood: {sum([x[1] for x in loglikelihoods]) / len(loglikelihoods)}')
    current_loglikelihood = sum([x[1] for x in loglikelihoods]) / len(loglikelihoods)
    global_best_loglikelihood = current_loglikelihood
    global_best_text = encoded_text.tolist()
    current_used_patience = 0
    try:
        while patience > current_used_patience:
            new_tokens_list = replace_characters(chars_map, loglikelihoods, num_to_replace=1)
            loglikelihoods = loglikelihood(new_tokens_list)
            current_loglikelihood = sum([x[1] for x in loglikelihoods]) / len(loglikelihoods)
            print(f'Mean loglikelihood: {current_loglikelihood}')
            print(f'New text: {decode_tokens(new_tokens_list)}')
            if current_loglikelihood < global_best_loglikelihood:
                global_best_loglikelihood = current_loglikelihood
                global_best_text = new_tokens_list.tolist()
                current_used_patience = 0
            else:
                current_used_patience += 1
    except ValueError:
        print('No more characters to replace.')
    
    # Reconstruct the text
    text = decode_tokens(global_best_text)
    return text

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
            if char in final_map.keys():
                rewritten_text += random.choice(final_map[char])
            else:
                # other type of character so write it to file as it is.
                rewritten_text += char
        rewritten_text += " "

    return rewritten_text

def rewrite_attack(text, replace_chars_fn=replace_characters_by_equivalents, do_replace_spaces=True):
    rewritten_text = text
    if replace_chars_fn is not None:
        rewritten_text = replace_characters_by_equivalents(chars_map, rewritten_text)
    if do_replace_spaces:
        rewritten_text = replace_spaces(rewritten_text)
    return rewritten_text

if __name__ == "__main__":
    # legitimate_text = input("Give the legitimate/original text to be perturbed in 1 line:\n")
    original_text = """The following is a transcript from The Guardian's interview with the British ambassador to the UN, John Baird. Baird: The situation in Syria is very dire. We have a number of reports of chemical weapons being used in the country. The Syrian opposition has expressed their willingness to use chemical weapons. We have a number of people who have been killed, many of them civilians. I think it is important to understand this. There are many who are saying that the chemical weapons used in Syria are not only used to destroy people but also to destroy the Syrian people. The Syrian people have been suffering for many years. The regime is responsible for that suffering. They have been using chemical weapons. They have killed many people, and they continue to kill many more. I think that the international community has to take a position that the Assad regime has a responsibility for that suffering. It must take a stand that we are not going to allow the Syrian government to use chemical weapons on civilians, that we are not going to allow them, and that we do not condone their use. We have a lot of people who believe that the regime is responsible for this suffering, and that they are responsible for this suffering, and that they are responsible for the use of chemical weapons. I think that we need to be clear about that. We must be clear that the use of chemical weapons by any country, including Russia and Iran, is a violation of international law. We are not going to tolerate that. We do not tolerate that. And we have the responsibility to ensure that the world doesn't allow the Assad regime to use chemical weapons against civilians. Baird: It seems that there are a range of people that are saying that we are not allowed to use chemical weapons in Syria. There are many who say we are not allowed to use chemical weapons in Syria. I think there are a lot of people that are saying that we are not allowed to use chemical weapons in Syria. I think that we have to take a stand that we are not going to allow the Assad regime to use chemical weapons on civilians, that we are not going to tolerate that. We have to take a stand that we are not going to allow Russia and Iran to use chemical weapons on civilians. Baird: I think it is important for us to understand that the use of chemical weapons in Syria is an extremely dangerous situation. I think there has been very little information from the UN that the regime has used any chemical weapons. We have not seen any evidence that they are using them. We have to understand that the use of chemical weapons is very dangerous."""
    original_text = """What are the standards required of offered properties? Properties need to be habitable and must meet certain health and safety standards, which the local authority can discuss with you. These standards have been agreed by the Department of Housing, Local Government and Heritage. The local authority will assess your property to make sure it meets the standards. If the property does not meet the standards, the local authority will explain why and can discuss what could be done to bring the property up to standard. Some properties may not be suitable for all those in need of accommodation, due to location or other reasons. However, every effort will be made by the local authority to ensure that offered properties are matched to appropriate beneficiaries."""
    #original_text = """The following is a transcript from The Guardian's interview."""

    rewritten_text = rewrite_attack(original_text, do_replace_chars=True, do_replace_spaces=False)
    print("\n========================\n")
    print(rewritten_text)
    print("\n========================\n")

    ## Tokenize the text using GPT2Tokenizer and print its decoded form
    #from transformers import AutoTokenizer
    #tok = AutoTokenizer.from_pretrained('t5-base')
    #print(tok.decode(tok.encode(rewritten_text)))

# %%
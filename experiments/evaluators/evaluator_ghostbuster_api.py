# %%
# Create a PreTrainedModel with a custom forward method
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Callable, List, Dict, Union
import requests
import re
import random
import time
import logging
import math


logger = logging.getLogger(__name__)


class GhostbusterAPIConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class GhostbusterAPIModel(PreTrainedModel):
    config_class = GhostbusterAPIConfig

    def __init__(self, config: GhostbusterAPIConfig):
        super().__init__(config)

    @property
    def device(self):
        return "cpu"

    def compute_score(self, texts: List[str]) -> List[float]:
        # use scrapy to scrape the response
        # Avoid being blocked by the website
        # use scrapy-fake-useragent middleware
        scores = []

        for text in texts:
            # Perform this cURL request:
            """
            curl 'https://ghostbuster-api-gm7xhurxnq-uw.a.run.app/' \
    -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
    -H 'accept-language: en-GB,en-US;q=0.9,en;q=0.8' \
    -H 'cache-control: no-cache' \
    -H 'content-type: application/x-www-form-urlencoded' \
    -H 'origin: https://ghostbuster-api-gm7xhurxnq-uw.a.run.app' \
    -H 'pragma: no-cache' \
    -H 'priority: u=0, i' \
    -H 'referer: https://ghostbuster-api-gm7xhurxnq-uw.a.run.app/' \
    -H 'sec-ch-ua: "Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"' \
    -H 'sec-ch-ua-mobile: ?0' \
    -H 'sec-ch-ua-platform: "macOS"' \
    -H 'sec-fetch-dest: iframe' \
    -H 'sec-fetch-mode: navigate' \
    -H 'sec-fetch-site: same-origin' \
    -H 'sec-fetch-user: ?1' \
    -H 'upgrade-insecure-requests: 1' \
    -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36' \
    --data-raw 'content=Text' \
            """
            # The text is encoded as a x-www-form-urlencoded string
            # The response is a part of the HTML response (const prediction = 0.16389074005313664;)
            # Use regex to extract the score
            import requests

            headers = {
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
                #'cache-control': 'no-cache',
                'content-type': 'application/x-www-form-urlencoded',
                'origin': 'https://ghostbuster-api-gm7xhurxnq-uw.a.run.app',
                #'pragma': 'no-cache',
                #'priority': 'u=0, i',
                'referer': 'https://ghostbuster-api-gm7xhurxnq-uw.a.run.app/',
                #'sec-ch-ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
                #'sec-ch-ua-mobile': '?0',
                #'sec-ch-ua-platform': '"macOS"',
                'sec-fetch-dest': 'iframe',
                'sec-fetch-mode': 'navigate',
                'sec-fetch-site': 'same-origin',
                #'sec-fetch-user': '?1',
                'upgrade-insecure-requests': '1',
                #'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            }

            data = {
                'content': text,
            }

            response = requests.post('https://ghostbuster-api-gm7xhurxnq-uw.a.run.app/', headers=headers, data=data)
            try:
                score = re.search(r"const prediction = (.*);", response.text).group(1)
                scores.append(float(score))
                # logger.info(f"Score for text: '''{text}'''\n\n==> {score}\n\n")
            except Exception as e:
                logger.exception(f"Exception occurred while processing text: '''{text}''', got response:  '''{response.text}'''. Request body was '''{response.request.body}'''")
                # Append nan if the score cannot be extracted
                scores.append(float("nan"))
            time.sleep(random.randint(1500, 9000) / 1000)
        return scores

    def forward(
        self, texts: List[str], rewriter: Optional[Callable[[str], str]], **kwargs
    ) -> List[Dict[str, Union[float, int]]]:
        if rewriter:
            if isinstance(texts, List):
                texts = [rewriter(text) for text in texts]
            else:
                texts = rewriter(texts)
        # Compute the score
        threshold = 0.5  # The threshold for the score in ghostbuster will be 50% generated - this is because we need to pick some threshold and this is a binary classifier between 0 and 1
        scores = self.compute_score(texts)
        # If the score is less than the threshold, the text is not generated
        # If the score is greater than the threshold, the text is generated
        # If it is not between 0 and 1, it is an error (nan), we should record nan
        generated = []
        for i, score in enumerate(scores):
            if math.isnan(score):
                generated.append(-1) # -1 is the code for error
            # 0 if score < threshold else 1 for score in scores
            elif score < threshold:
                generated.append(0)
            else:
                generated.append(1)
        return [
            {"score": score, "generated": gen} for score, gen in zip(scores, generated)
        ]

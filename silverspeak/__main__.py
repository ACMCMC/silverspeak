import logging
import random

from silverspeak.homoglyphs.random_attack import random_attack

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(42)
original_text = """The following is a transcript from The Guardian's interview with the British ambassador to the UN, John Baird.

Baird: The situation in Syria is very dire. We have a number of reports of chemical weapons being used in the country. The Syrian opposition has expressed their willingness to use chemical weapons. We have a number of people who have been killed, many of them civilians. I think it is important to understand this.

There are many who are saying that the chemical weapons used in Syria are not only used to destroy people but also to destroy the Syrian people. The Syrian people have been suffering for many years. The regime is responsible for that suffering. They have been using chemical weapons. They have killed many people, and they continue to kill many more.

I think that the international community has to take a position that the Assad regime has a responsibility for that suffering. It must take a stand that we are not going to allow the Syrian government to use chemical weapons on civilians, that we are not going to allow them, and that we do not condone their use.

We have a lot of people who believe that the regime is responsible for this suffering, and that they are responsible for this suffering, and that they are responsible for the use of chemical weapons. I think that we need to be clear about that.

We must be clear that the use of chemical weapons by any country, including Russia and Iran, is a violation of international law. We are not going to tolerate that. We do not tolerate that. And we have the responsibility to ensure that the world doesn't allow the Assad regime to use chemical weapons against civilians.

Baird: It seems that there are a range of people that are saying that we are not allowed to use chemical weapons in Syria. There are many who say we are not allowed to use chemical weapons in Syria.

I think there are a lot of people that are saying that we are not allowed to use chemical weapons in Syria. I think that we have to take a stand that we are not going to allow the Assad regime to use chemical weapons on civilians, that we are not going to tolerate that. We have to take a stand that we are not going to allow Russia and Iran to use chemical weapons on civilians.

Baird: I think it is important for us to understand that the use of chemical weapons in Syria is an extremely dangerous situation. I think there has been very little information from the UN that the regime has used any chemical weapons. We have not seen any evidence that they are using them.

We have to understand that the use of chemical weapons is very dangerous."""

original_text = """What are the standards required of offered properties? Properties need to be habitable and must meet certain health and safety standards, which the local authority can discuss with you. These standards have been agreed by the Department of Housing, Local Government and Heritage. The local authority will assess your property to make sure it meets the standards. If the property does not meet the standards, the local authority will explain why and can discuss what could be done to bring the property up to standard. Some properties may not be suitable for all those in need of accommodation, due to location or other reasons. However, every effort will be made by the local authority to ensure that offered properties are matched to appropriate beneficiaries."""
original_text = """What standards are required for offered properties? Properties must be habitable and meet certain health and safety standards, which can be discussed with the local authority. These standards have been agreed upon by the Department of Housing, Local Government and Heritage. The local authority will assess your property to ensure it meets the standards. If the property does not meet the required standards, the local authority will provide an explanation and discuss potential solutions to bring the property up to standard. Some properties may not be suitable for all individuals seeking accommodation due to location or other factors. However, the local authority will make every effort to match offered properties with appropriate beneficiaries."""

rewritten_text = random_attack(original_text, percentage=0.15)
logger.info("\n========================\n")
logger.info(rewritten_text)
logger.info("\n========================\n")

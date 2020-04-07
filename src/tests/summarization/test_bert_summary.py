import unittest

from summarization.bert_summary import BertSummarizer


class TestBertSummarizer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert_summarizer = BertSummarizer()
        self.sentences = [
            "The virus is transmitted through droplets, close contact, and other means , and patients in the "
            "incubation period could potentially transmit the virus to other persons.",
            "Disease spread through both direct (droplet and person-to-person) as well as indirect contact ("
            "contaminated objects and airborne transmission) are indicated , supporting the use of airborne isolation "
            "precautions.",
            "Analysis indicated virus transmission from a traveler from china .",
            "here we highlight nine most important research questions concerning virus transmission , asymptomatic "
            "and presymptomatic virus shedding, diagnosis, treatment, vaccine development, origin of virus and viral "
            "pathogenesis.",
            "We read with interest recent article by zhang et al (1) on the diagnosis of coronavirus disease 2019 ("
            "covid-19 ) by fecal specimen test .",
            "Conclusions : our results demonstrated the presence of sars-cov-2 rna in feces of covid-19 patients, "
            "and suggested the possibility of sars-cov-2 transmission via the fecal-oral route .",
            "Although coronaviruses usually infect the upper or lower respiratory tract, viral shedding in plasma or "
            "serum is common. therefore, there is still a theoretical risk of transmission of coronaviruses through "
            "the transfusion of labile blood products .",
            "Severity of disease was mild to moderate and fever was the most consistent and predominant symptom at "
            "onset of illness of these children (two cases had body temperature higher than 38. 5°c). all children "
            "showed increased lymphocytes (> 4. 4×10 9 / l) with normal white blood cell counts on admission. "
            "radiological changes were not typical for covid-19. all children showed good response to supportive "
            "treatment. clearance of sars-cov-2 in respiratory tract occurred within two weeks after abatement of "
            "fever, whereas viral rna remained positive in stools of pediatric patients for longer than 4 weeks. two "
            "children had fecal sars-cov-2 turned negative 20 days after throat swabs showing negative, while that of "
            "another child lagged behind for 8 days. interpretation : sars-cov-2 may exist in gastrointestinal tract "
            "for a longer time than respiratory system. persistent shedding of sars-cov-2 in stools of infected "
            "children indicates the potential for the virus to be transmitted through fecal excretion .",
            "The new coronavirus was reported to spread via droplets, contact and natural aerosols from "
            "human-to-human .",
            "Data concerning the transmission of the novel severe acute respiratory syndrome coronavirus (sars-cov-2) "
            "in paucisymptomatic patients are lacking .",
            "There is a new public health crises threatening the world with the emergence and spread of 2019 novel "
            "coronavirus (2019-ncov)",
        ]

    def test_bert_summarizer(self):
        text = ' '.join(self.sentences)
        summary = self.bert_summarizer.create_summary(text)
        print(summary)
        assert summary == "There is a new public health crises threatening the world with the emergence and " \
                          "spread of 2019 novel coronavirus (2019-ncov) The new coronavirus was reported to spread " \
                          "via droplets, contact and natural aerosols from human-to-human. The virus is transmitted " \
                          "through droplets, close contact, and other means. Patients in the incubation period " \
                          "could potentially transmit the virus to other persons."

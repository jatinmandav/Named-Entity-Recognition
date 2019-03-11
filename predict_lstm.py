import fasttext
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
import numpy as np
import pickle
from keras.models import load_model

sentTokenizer = PunktSentenceTokenizer()
max_len = 50

def preprocess(text):
    sentences = sentTokenizer.tokenize(text)
    sentences = [word_tokenize(sent) for sent in sentences]

    padded_X = []
    for seq in sentences:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append('__PAD__')

        padded_X.append(new_seq)

    return padded_X

def get_embeddings(x):
    embeddings = []
    for word in x:
        embeddings.append(skipgram_model[word])

    return np.array(embeddings)

text = """Democratic Party (United States)   From Wikipedia, the free encyclopedia    Jump to navigation  Jump to search  For other uses, see Democratic Party (disambiguation) .    political party in the United States    Democratic Party Chairperson Tom Perez ( MD ) Secretary Jason Rae ( WI ) Speaker of the House Nancy Pelosi ( CA ) House Majority Leader Steny Hoyer ( MD ) Senate Minority Leader Chuck Schumer ( NY ) Founded January\xa08, 1828 ; 191 years ago ( 1828-01-08 ) [1] Preceded\xa0by Democratic-Republican Party Headquarters 430 South Capitol St. SE, Washington, D.C. , 20003 Student wing College Democrats of America High School Democrats of America Youth wing Young Democrats of America Women\'s wing National Federation of Democratic Women Overseas wing Democrats Abroad LGBTQ wing Stonewall Democrats Membership\xa0(2017) 44,706,349 [2] Ideology Majority : • Modern liberalism [3] [4] • Social liberalism [5] Factions : • Centrism [6] [7] • Conservatism [8] [7] • Democratic socialism [9] [10] [11] [12] • Left-wing populism [13] [14] [15] • Progressivism [16] • Social democracy [17] Colors   Blue Seats in the Senate 45 / 100 Seats in the House 235 / 435 State Governorships 23 / 50 State Upper House Seats 874 / 1,972 State Lower House Seats 2,580 / 5,411 Total State Legislature Seats 3,454 / 7,366 Territorial Governorships 4 / 6 Territorial Upper Chamber Seats 31 / 97 Territorial Lower Chamber Seats 0 / 91 Website democrats.org Politics of United States Political parties Elections  This article is part of a series on the Politics of the United States of America  Federal Government  Constitution of the United States  Law  Taxation   Legislature  United States Congress        House of Representatives  Speaker  Nancy Pelosi (D)  Majority Leader  Steny Hoyer (D)  Minority Leader  Kevin McCarthy (R)  Congressional districts     Senate  President  Mike Pence (R)  President Pro Tempore  Chuck Grassley (R)  President Pro Tempore Emeritus  Patrick Leahy (D)  Majority Leader  Mitch McConnell (R)  Minority Leader  Chuck Schumer (D)   Executive  President of the United States  Donald Trump (R)    Vice President of the United States  Mike Pence (R)   Cabinet  Federal agencies  Executive Office   Judiciary  Supreme Court of the United States  Chief Justice  John Roberts  Thomas  Ginsburg  Breyer  Alito  Sotomayor  Kagan  Gorsuch  Kavanaugh   Courts of Appeals  District Courts  ( list )  Other tribunals   Elections  Presidential elections  Midterm elections  Off-year elections   Political parties  Democratic  Republican   Third parties  Libertarian  Green   Federalism  State Government   Governors  Legislatures ( List )  State courts  Local government    United States portal   Other countries  Atlas  v t e  The Democratic Party is one of the two  major contemporary political parties in the United States , along with the Republican Party . Tracing its heritage back to Thomas Jefferson and James Madison \'s Democratic-Republican Party , the modern-day Democratic Party was founded around 1828 by supporters of Andrew Jackson , making it the world\'s oldest active political party. [18] The Democrats\' dominant worldview was once social conservatism and economic liberalism while populism was its leading characteristic in the rural South . In 1912 , Theodore Roosevelt ran as a third-party candidate in the Progressive ("Bull Moose") Party , beginning a switch of political platforms between the Democratic and Republican Party over the coming decades, and leading to Woodrow Wilson being elected as the first fiscally progressive Democrat. Since Franklin D. Roosevelt and his New Deal coalition in the 1930s, the Democratic Party has also promoted a social liberal platform, [3] supporting social justice . [19]  Today, the House Democratic caucus is composed mostly of centrists and progressives , [6] with a small minority of conservative Democrats . The party\'s philosophy of modern liberalism advocates social and economic equality , along with the welfare state . [20] It seeks to provide government intervention and regulation in the economy. [21] These interventions, such as the introduction of social programs , support for labor unions , affordable college tuitions , moves toward universal health care and equal opportunity , consumer protection and environmental protection form the core of the party\'s economic policy. [20] [22] The party has united with smaller liberal regional parties throughout the country, such as the Farmer–Labor Party in Minnesota and the Nonpartisan League in North Dakota . Well into the 20th century, the party had conservative pro-business and Southern conservative-populist anti-business wings. The New Deal Coalition of 1932–1964 attracted strong support from voters of recent European extraction—many of whom were Catholics based in the cities. [23] [24] [25] After Franklin D. Roosevelt\'s New Deal of the 1930s, the pro-business wing withered outside the South. After the racial turmoil of the 1960s, most Southern whites and many Northern Catholics moved into the Republican Party at the presidential level. The once-powerful labor union element became smaller and less supportive after the 1970s. White Evangelicals and Southerners became heavily Republican at the state and local level since the 1990s. People living in metropolitan areas , women, sexual minorities , millennials , college graduates, and racial and ethnic minorities [26] in the United States, such as Jewish Americans , Hispanic Americans , Asian Americans , [27]  Arab Americans and African Americans , tend to support the Democratic Party much more than they support the rival Republican Party. Fifteen Democrats have served as President under sixteen administrations: the first was seventh President Andrew Jackson , who served from 1829 to 1837; Grover Cleveland served two nonconsecutive terms from 1885 to 1889 and 1893 to 1897; and thus is counted twice (as the twenty-second and twenty-fourth President). The most recent was the forty-fourth President Barack Obama , who held the office from 2009 to 2017."""

X = preprocess(text)
skipgram_model = fasttext.load_model('skipgram.bin')

sent_embedding = []
for sent in X:
    sent_embedding.append(get_embeddings(sent))

sent_embedding = np.array(sent_embedding)
print(sent_embedding.shape)

lstm_model = load_model('logs/final-model.h5')
pred = lstm_model.predict(sent_embedding)
print(pred.shape)

tag2index = pickle.load(open('tag2index.pkl', 'rb'))

print('Word, Tag')

for i, sent in enumerate(pred):
    for j, word in enumerate(sent):
        index = np.argmax(word)
        if X[i][j] != "__PAD__":
            print(X[i][j], tag2index[index])

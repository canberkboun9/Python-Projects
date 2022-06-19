import string as Stringg
import json


json1_file = open('sample.json')
json1_str = json1_file.read()
inverted_index = json.loads(json1_str)


def intersect(word1, word2):
    answer = []
    len_1 = len(word1)
    len_2 = len(word2)
    i = 0
    j = 0
    while i < len_1 - 1 and j < len_2 - 1:
        if int(word1[i]) == int(word2[j]):
            answer.append(int(word1[i]))
            i+=1
            j+=1
        elif int(word1[i]) < int(word2[j]):
            i+=1
        else:
            j+=1

    return answer


# Main
is_continue = "continue"
while is_continue != "q" and is_continue != "Q":
    print("Please enter the query: ")
    x = input()

    query = x.split()
    words = []
    not_words = []
    result = []

    if "AND" in query:
        if "NOT" in query:  # Conjunction and Negation
            index_of_not = query.index("NOT")

            for word in query[index_of_not:]:
                if word != "NOT":
                    not_words.append(word.lower())

            for word in query[:index_of_not]:
                if word != "AND":
                    words.append(word.lower())

            not_found = True
            while not_found:
                if words[0] in inverted_index:
                    result = inverted_index[words[0]]
                    not_found = False
                else:
                    if len(words) < 2:
                        not_found = False
                        continue
                    else:
                        words = words[1:]

            for i in range(len(words) - 1):
                if words[i + 1] in inverted_index:  # to avoid key errors.
                    result = intersect(result, inverted_index[words[i + 1]])  # calculated the and part. go to not part.

            # If we remove the articles containing the keywords in not_words list from the result list,
            # we would find the result of the Conjunction and Negation query.
            for word in not_words:
                if word in inverted_index:
                    list_of_articles = inverted_index[word]
                    for article in list_of_articles:
                        if int(article) in result:
                            result.remove(int(article))

            print("Result: ", result)

        else:  # Conjunction
            for word in query:
                if word != "AND":
                    words.append(word.lower())  # keywords are captured. go on and make the AND query

            not_found = True
            while not_found:
                if words[0] in inverted_index:
                    result = inverted_index[words[0]]
                    not_found = False
                else:
                    if len(words)<2:
                        not_found = False
                        continue
                    else:
                        words = words[1:]

            for i in range(len(words) - 1):
                if words[i+1] in inverted_index:  # to avoid key errors.
                    result = intersect(result, inverted_index[words[i+1]])

            print("Result: ", result)

    else:
        if "NOT" in query:  # Disjunction and Negation
            index_of_not = query.index("NOT")

            for word in query[index_of_not:]:
                if word != "NOT":
                    not_words.append(word.lower())

            for word in query[:index_of_not]:
                if word != "OR":
                    words.append(word.lower())  # keywords are captured. go on and make the OR query

            # We don't need the merge algorithm here.
            for word in words:
                if word in inverted_index:  # to avoid key errors.
                    result.extend(list(set(inverted_index[word])))  # to avoid duplicates.

            # If we remove the articles containing the keywords in not_words list from the result list,
            # we would find the result of the Conjunction and Negation query.
            for word in not_words:
                if word in inverted_index:
                    list_of_articles = inverted_index[word]
                    for article in list_of_articles:
                        if int(article) in result:
                            result.remove(int(article))

            print("Result: ", result)
        else:  # Full of OR query. Disjunction
            for word in query:
                if word != "OR":
                    words.append(word.lower())  # keywords are captured. go on and make the OR query
            # We don't need the merge algorithm here.
            for word in words:
                if word in inverted_index:
                    result.extend(list(set(inverted_index[word])))  # to avoid duplicates.


            print("Result: ", result)

    print("If you want to quit, enter the letter 'Q', If you want to continue, enter another letter: ")
    is_continue = input()

print("Have a nice day! Thank you for using the query processor master99.")


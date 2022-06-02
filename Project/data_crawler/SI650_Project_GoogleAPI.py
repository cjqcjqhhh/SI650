import csv
import requests

KEY = "AIzaSyBu5hG9F7SbwrNn2DvU7y35MB9lpQltGwg"


def findValue(jsonFile, key):
    if key in jsonFile:
        return jsonFile[key]
    else:
        return None


with open('documents.csv', 'r', newline='') as iFile, open('tourist_attractions.csv', 'a', newline='') as oFile:
    csvreader = csv.reader(iFile, delimiter=',', quotechar='"')
    csvwriter = csv.writer(oFile, delimiter=',', quotechar='"')
    header = next(csvreader)
    header += ['map_name', 'place_id', 'lat', 'lng', 'rating', 'review_text_1', 'review_time_1', 'review_text_2',
               'review_time_2', 'review_text_3', 'review_time_3', 'review_text_4', 'review_time_4', 'review_text_5',
               'review_time_5']
    csvwriter.writerow(header)
    for row in csvreader:
        place = row[0]
        state = row[1]
        text = place.replace(' ', '%20')
        text += "%20" + state

        url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input=" + text + "&inputtype" \
                                                                                                  "=textquery&fields" \
                                                                                                  "=name%2Cgeometry" \
                                                                                                  "%2Crating" \
                                                                                                  "%2Cplace_id&key" \
                                                                                                  "=" + KEY
        payload = {}
        headers = {}

        response = requests.request("GET", url, headers=headers, data=payload)
        jsonFile = response.json()
        print(response.text)
        isFound = False
        name = ""
        place_id = ""
        geometry = ""
        rating = ""
        if jsonFile["status"] == "OK":
            print(response.text)
            candidate = jsonFile["candidates"][0]
            name = candidate["name"]
            place_id = candidate["place_id"]
            geometry = candidate["geometry"]
            rating = findValue(candidate, "rating")
            isFound = True

        if isFound:
            row += [name, place_id, geometry["location"]["lat"], geometry["location"]["lng"], rating]
            url = "https://maps.googleapis.com/maps/api/place/details/json?place_id=" + place_id + "&fields=reviews" \
                                                                                                   "&key=" + KEY
            response = requests.request("GET", url, headers=headers, data=payload)
            jsonFile = response.json()
            print(response.text)
            if jsonFile["status"] == "OK":
                result = jsonFile["result"]
                reviews = findValue(result, "reviews")
                if reviews:
                    for review in reviews:
                        row += [review["text"], review["time"]]

        csvwriter.writerow(row)



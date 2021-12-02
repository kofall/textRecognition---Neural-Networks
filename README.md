# textRecognition---Neural-Networks
Program that reads text from picture using Neural Networks.

Na początku program tworzy obiekt, do którego podajemy wczytane zdjęcie. Obiekt przetwarza go na różne sposoby.
Ostatecznie umożliwia on zwrócenie takich zdjęć jak:
  getImage() - zwraca aktualnie przetwarzane zdjęcie
  getGrayedImage() - zwraca zdjęcie przekonwertowane do skali szarości
  getCanniedImage() - zwraca rezultat działania algorytmu Canniego do wykrywania krawędzi
  getContouredImage() - zwraca zdjęcie z narysowanymi konturami
  getContouredImageWithBox() - zwraca zdjęcie z narysowanymi konturami oraz prostokątem, otaczającym kontur
  getCutImages() - zwraca tablicę wykrytych zdjęć liter

Zwrócona tablica liter odczytywana jest przez drugi wątek, który stara się wywnioskować, na podstawie wygenerowanego wcześniej modelu, jaka litera widnieje na zdjęciu. 

Ponadto w programie możliwe jest dopasowanie parametrów by odpowiadały one aktualnie przetwarzanemu zdjęciu. Są to m.in.:
  - SOURCE -> sposób przetwarzania zależy od wybooru źródła zdjęcia:
    - 0 -> zdjęcie wykonane, np. w paint'cie
    - 1 -> zdjęcie wykonane za pomocą aparatu
    - 2 -> kamera
  - TH1 i TH2 -> progi koloru
  - MIN AREA -> odszumienie zdjęcia poprzez wykrycie konturu o odpowiedniej powierzchni
  - ROW APPROX -> rozdzielenie rzędów poprzez przyjęcie zakresu, w którym wykrywany jest kontur
  - MIN GAP -> rozdzielenie wyrazów poprzez przyjęcię minimalnej przerwy jako spacji pomiędzy kolejnymi literami
  - BACKGROUND APPROX -> skalowanie litery
  - BOX APPROX -> skalowanie wielkości wyciętej litery ze zdjęcia
Funkcja updateStatus() aktualizuje powyższe parametry.

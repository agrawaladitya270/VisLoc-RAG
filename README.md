# Visual + Location Aware RAG
   <p align="center">
  <img src="https://github.com/user-attachments/assets/8ae21712-2b9a-475c-a548-0a1682540660" alt="Visual + Location Aware RAG" />
</p>                                       

This project implements a retrieval pipeline that combines image understanding and location context to improve information retrieval. It is designed to solve a common problem in visual search and retrieval-augmented generation (RAG): generic image recognition often produces ambiguous or incorrect results when the same object or landmark exists in multiple locations.

## Motivation

Whenever I go hiking or visit historical places in Japan, I like to learn about the history, stories, and interesting facts behind those places. However, most of the useful information is available in Japanese, and manually searching and translating it every time is inconvenient.

This project was built to solve that problem. It identifies what I’m looking at from an image, combines it with my location, searches relevant local sources, and provides meaningful information in English. This makes it easy to explore and learn about places without manually searching or translating content.

## Solution

This project uses a location-first approach:

1. The system accepts a known location (currently manual input).
2. The uploaded image is analyzed by a vision model to identify the subject.
3. The system combines:

   * Visual description (what is in the image)
   * Location (where the image was taken)
4. A combined query is generated using both signals.
5. Local-language web search is performed using the combined query.
6. Retrieved content is translated into English (if needed).
7. The results are passed into a RAG pipeline for answering or summarization.

In future versions, GPS-based location detection will be added to automatically detect the user’s location without manual input.

## Example Flow

Input:

* Image of a gate or landmark
* Location: “Tokyo, Japan”

Process:

* Vision model identifies: “Kaminarimon Gate”
* Combined query: “Kaminarimon Gate Tokyo history”
* Local web results retrieved (Japanese sources)
* Translated and passed into RAG

Output:

* Accurate, location-specific information about the landmark.

## Features

* Location-first retrieval strategy
* Image-aware query generation
* Local web search without paid APIs
* Optional multilingual handling
* Simple integration with RAG systems
* Planned support for GPS-based automatic location detection

## Requirements

* Python 3.9+
* Dependencies:

  * openai
  * requests
  * beautifulsoup4
  * deep-translator (optional)

## Usage

1. Provide a location (currently manual).
2. Provide an image.
3. Run the notebook.
4. Receive context grounded in both visual and geographic information.

## Use Cases

* Travel and tourism apps
* AR experiences
* Cultural heritage exploration
* Visual question answering with geographic grounding
* Location-aware chat assistants

## Limitations

* Visual recognition depends on the quality of the input image.
* Web results depend on search engine availability.
* Translation quality may vary for niche topics.
* GPS-based location detection is not yet implemented.

import logging
import googlemaps

from typing import List
from pydantic import BaseModel, Field
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import PromptTemplate

class AgentMapSearchRec(object):
    def __init__(self, GROQ_API_KEY=None, OPENAI_API_KEY=None, GOOGLE_MAPS_API_KEY=None):
        """
        Initialize the AgentMapSearchRec object with API keys.

        Parameters:
        ----------
        groq_api_key : str, optional
            API key for the GROQ LLM service.
        openai_api_key : str, optional
            API key for OpenAI.
        google_maps_api_key : str, optional
            API key for Google Maps.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._groq_key = GROQ_API_KEY
        self._openai_key = OPENAI_API_KEY
        self._google_maps_key = GOOGLE_MAPS_API_KEY
        self.gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        if GROQ_API_KEY is not None:
            self.llm = Groq(model="llama-3.1-70b-versatile", api_key=GROQ_API_KEY)
        elif OPENAI_API_KEY is not None:
            self.llm = OpenAI(model="gpt-4o-mini",api_key = OPENAI_API_KEY)
        Settings.llm = self.llm
    
    def get_top_recommendations(self, user_search_request_str):
        """

        Parameters
        ----------
        user_search_request_str : str
            User search request.

        Returns
        -------
        recommendations_output : list
            list of objects with recommended place information: name, google maps url, summary

        """
        search_output_specifics = self.parse_user_search_request(user_search_request_str)
        places = self.gmaps.places(
            search_output_specifics.location + ' ' + search_output_specifics.search_type + ' ' + search_output_specifics.criteria
            )
        place_id_ls = [place['place_id'] for place in places['results']]
        place_info_ls = [self._get_place_id_info(place_id) for place_id in place_id_ls]

        class Recommendation(BaseModel):
            """Object representing a single recommendation."""

            name: str = Field(description="Name of the recommended place.")
            photo_url: List = Field(description="List of photo urls of the recommended place.")
            maps_location_url: str = Field(description="Google maps location url")
            selection_reason: str = Field(description="Selection reason and fulfilment of user criteria")
            summary: str = Field(description="In 5 bulletins, give short description of the place from reviews and details such as opening hours, price, ratings, number of review.")

        class Recommendations(BaseModel):
            """Object representing a list of recommendations."""

            recommendations: List[Recommendation] = Field(description="List of recommendations. Not more than 5.")

        search_results_criteria_prompt_str = """\
        You are a travel agent who helps provide recommendations from user criteria and google map search results.
        The user criteria and google map search results will be denoted by four hashtags. Parse the information into a list of recommendations, ranked accordingly.
        ####{query}####
        """
        recommendations_parser = PydanticOutputParser(Recommendations)
        search_results_criteria_json_prompt_str = recommendations_parser.format(search_results_criteria_prompt_str)
        search_results_criteria_prompt_tmpl = PromptTemplate(search_results_criteria_json_prompt_str)
        p = QueryPipeline(chain=[search_results_criteria_prompt_tmpl, self.llm, recommendations_parser], verbose=True)
        query = f"user criteria: {search_output_specifics.criteria} + ' ' search results: {str(place_info_ls)}" 
        recommendations_output = p.run(query=query)

        return recommendations_output.recommendations

    def _get_place_id_info(self, place_id):
        """

        Parameters
        ----------
        place_id : str
            The google place id of a location.

        Returns
        -------
        place_info : dict
            The information of a place id such as price, rating, review, google maps url in a dictionary.

        """

        place_info = {}
        place_info_raw = self.gmaps.place(place_id)
        for key in ['name','formatted_address','price_level','rating','user_ratings_total','url']:
                if key in place_info_raw['result']:
                        place_info[key] = place_info_raw['result'][key]
                else:
                        place_info[key] = None
        try:
                place_info['opening_hours'] = place_info_raw['result']['current_opening_hours']['weekday_text']
        except:
                place_info['opening_hours'] = None
        try:
                photo_reference_0 = place_info_raw['result']['photos'][0]['photo_reference']
                photo_reference_1 = place_info_raw['result']['photos'][1]['photo_reference']
                photo_reference_2 = place_info_raw['result']['photos'][2]['photo_reference']
                photo_url_0 = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference_0}&key={self._google_maps_key}"
                photo_url_1 = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference_1}&key={self._google_maps_key}"
                photo_url_2 = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference_2}&key={self._google_maps_key}"
                place_info['photo_url'] = [photo_url_0, photo_url_1, photo_url_2]
        except:
                place_info['photo_url'] = None

        return place_info
    
    def parse_user_search_request(self, user_search_request_str):
        """

        Parameters
        ----------
        user_search_request_str : str
            User search request.

        Returns
        -------
        search_output_specifics : object
            Object parsed with search specifics of location, search_type, criteria.

        """
        class RequestSpecifics(BaseModel):
            location: str = Field(description="area or place or town or city of user request")
            search_type: str = Field(description="type of search such as food, accomodation, etc.")
            criteria: str = Field(description="criteria for the search, such as budget, outdoors, western cuisine, swimming pool, etc.")
        
        parse_user_request_prompt_str = """\
        You are a travel agent who helps parse user request for google map searches.
        The user's itinerary will be denoted by four hashtags. Parse user request to its location, search type and criteria.
        ####{user_search_request_str}####
        """
        user_search_request_parser = PydanticOutputParser(RequestSpecifics)
        search_request_json_prompt_str = user_search_request_parser.format(parse_user_request_prompt_str)
        search_request_prompt_tmpl = PromptTemplate(search_request_json_prompt_str)
        p = QueryPipeline(chain=[search_request_prompt_tmpl, self.llm, user_search_request_parser], verbose=True)
        search_output_specifics = p.run(query=user_search_request_str)
        self.logger.info("Parse user request")
        self.logger.info(search_output_specifics)
        return search_output_specifics

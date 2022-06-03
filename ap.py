import character_recog
import digit_recog
import home_page
import traffic_signs
import fashion
import flowers
import earth
import streamlit as st


pages={"home_page":home_page,
       "Types of flowers":flowers,
       "places on earth":earth,
       "Different types of clothes":fashion,
       "traffic_signs":traffic_signs,
       "Alphabet characters":character_recog,
       "digits":digit_recog
       }


st.sidebar.title('IMAGE CLASSIFICATION')
selection = st.sidebar.radio("Choose from",list(pages.keys()))
page = pages[selection]
page.app()
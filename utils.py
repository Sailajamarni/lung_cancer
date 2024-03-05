import streamlit as st
from streamlit.components.v1 import html
from PATHS import NAVBAR_PATHS

def inject_custom_css():
    with open('assets/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        

def get_current_route():
    try:
        return st.experimental_get_query_params()['nav'][0]
    except:
        return None

def navbar_component():
    navbar_items = ''
    for key, value in NAVBAR_PATHS.items():
        navbar_items += (f'<a class="navitem" href="/?nav={value}">{key}</a>')
    
    component = rf'''
            <nav class="container navbar" id="navbar">
                <ul class="navlist">
                {navbar_items}
                </ul>
                
            </nav>
            '''
    st.markdown(component, unsafe_allow_html=True)
    js = '''
    <script>
        var navigationTabs = window.parent.document.getElementsByClassName("navitem");
        var cleanNavbar = function(navigation_element) {
            navigation_element.removeAttribute('target')
        }
        
        for (var i = 0; i < navigationTabs.length; i++) {
            cleanNavbar(navigationTabs[i]);
        }
    </script>
    '''
    html(js)
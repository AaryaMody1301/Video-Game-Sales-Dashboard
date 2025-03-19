"""
Callbacks for data export functionality
"""
from dash.dependencies import Input, Output, State
from dash import dcc
import pandas as pd
from io import StringIO, BytesIO
import base64
from src.data.data_loader import apply_filters

def register_export_callbacks(app, df, df_cache):
    """
    Register callbacks for data export functionality
    
    Args:
        app (dash.Dash): The Dash application
        df (pandas.DataFrame): The complete dataframe
        df_cache (DataFrameCache): Cache for filtered dataframes
    """
    @app.callback(
        Output("download-dataframe-csv", "data"),
        [Input("export-button", "n_clicks"),
         Input("export-format-dropdown", "value")],
        [State('year-slider', 'value'),
         State('platform-dropdown', 'value'),
         State('console-gen-dropdown', 'value'),
         State('genre-dropdown', 'value'),
         State('publisher-dropdown', 'value'),
         State('critic-score-slider', 'value'),
         State('search-bar', 'value')],
        prevent_initial_call=True,
    )
    def export_data(n_clicks, export_format, year_range, selected_platforms, selected_generations, selected_genres, 
                   selected_publishers, critic_range, search_value):
        """
        Export the filtered data in various formats
        
        Returns:
            dict: Dictionary containing the file content and metadata
        """
        if n_clicks is None:
            return None
            
        # Get the filtered dataframe
        filtered_df = apply_filters(df, df_cache, year_range, selected_platforms, selected_generations, selected_genres, 
                                   selected_publishers, critic_range, search_value)
        
        # Generate a filename with date
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_game_sales_{date_str}"
        
        # Export based on selected format
        if export_format == 'csv':
            return dcc.send_data_frame(filtered_df.to_csv, f"{filename}.csv", index=False)
        elif export_format == 'excel':
            # For Excel, we need to create a BytesIO object
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='VideoGameSales')
            excel_data = output.getvalue()
            return dcc.send_bytes(excel_data, f"{filename}.xlsx")
        elif export_format == 'pdf':
            # PDF export requires additional libraries like reportlab or pdfkit
            # Here, we'll use a simple HTML-to-PDF approach
            try:
                import pdfkit
                html_io = StringIO()
                filtered_df.to_html(html_io)
                html_str = html_io.getvalue()
                pdf = pdfkit.from_string(html_str, False)
                return dcc.send_bytes(pdf, f"{filename}.pdf")
            except ImportError:
                # If pdfkit is not available, fallback to CSV
                return dcc.send_data_frame(filtered_df.to_csv, f"{filename}.csv", index=False)
        
        return None 
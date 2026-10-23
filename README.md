# DK1 wind and solar forecasting

How close can a forecast of DK1 wind and solar power, made before the noon day-ahead auction from public weather forecasts, get to what is produced? Energinet publishes its own forecast only after the auction, so a trader can't use it to bid. In Python, with the data in DuckDB.

## Setup

- **Targets:** DK1 onshore wind, offshore wind and solar per hour, from Energinet's settlement data. Three separate models.
- **When:** each forecast is made at 10:00 Danish time for the 24 hours of the next day.
- **Weather:** Open-Meteo forecasts made two days before each hour, at nine points over DK1's wind farms and solar areas. That way they were out before the auction.
- **Known output:** settled production comes 8-10 days late, so a forecast can only use output up to 10 days before the day it forecasts.
- **Fleet size:** the highest hourly output in the 60 days up to that point, per target. The models forecast output as a share of it, so new parks (such as Thor offshore) don't break them.

## License

Code: [MIT](LICENSE). Data from [Energi Data Service](https://www.energidataservice.dk) (Energinet) and [Open-Meteo.com](https://open-meteo.com) ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).

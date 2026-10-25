# DK1 wind and solar forecasting

How close can a forecast of DK1 wind and solar power, made before the noon day-ahead auction from public weather forecasts, get to what is produced? Energinet publishes its own forecast only after the auction, so a trader can't use it to bid. In Python, with the data in DuckDB.

## Setup

- **Targets:** DK1 onshore wind, offshore wind and solar per hour, from Energinet's settlement data. Three separate models.
- **When:** each forecast is made at 10:00 Danish time for the 24 hours of the next day.
- **Weather:** Open-Meteo forecasts made two days before each hour, at nine points over DK1's wind farms and solar areas. That way they were out before the auction.
- **Known output:** settled production comes 8-10 days late, so a forecast can only use output up to 10 days before the day it forecasts.
- **Fleet size:** the highest hourly output in the 60 days up to that point, per target. The models forecast output as a share of it, so new parks (such as Thor offshore) don't break them.

## Rules, set before any model is scored

- **Backtest:** from March 2025 to the latest settled day. The models are refit on the first of each month on all data up to 10 days before.
- **Models:** a power curve (average share of the fleet in each bin of forecast wind speed or radiation), LightGBM on all weather points plus hour and day of the year, and the average of the two.
- **Score:** mean absolute error in MW, and as a share of the fleet. All hours count, including hours where parks turned down at negative prices.
- **Worth it:** LightGBM is worth it if its daily mean absolute error is lower than the power curve's, Diebold-Mariano with Newey-West, p < 0.05.
- **Energinet:** its day-ahead forecast is reported but isn't pass/fail. It uses later weather, and it doesn't seem to cover all of DK1's offshore fleet.

## License

Code: [MIT](LICENSE). Data from [Energi Data Service](https://www.energidataservice.dk) (Energinet) and [Open-Meteo.com](https://open-meteo.com) ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).

from src.constants import SUBMISSION_FORMAT_FLIGHT_ID, COLUMN_NAME_TIMESTAMP, SUBMISSION_FORMAT_AIRPORT


def rearrange_submission(rearranger, slutty):
    return (
        slutty.set_index([
            SUBMISSION_FORMAT_FLIGHT_ID,
            COLUMN_NAME_TIMESTAMP,
            SUBMISSION_FORMAT_AIRPORT
        ])
        .loc[
            rearranger.set_index([
                SUBMISSION_FORMAT_FLIGHT_ID,
                COLUMN_NAME_TIMESTAMP,
                SUBMISSION_FORMAT_AIRPORT
            ]).index
        ]
        .reset_index()
    )
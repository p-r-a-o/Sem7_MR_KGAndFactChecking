"""Uniform API errors: {"error": {"code": ..., "message": ...}} with a matching HTTP status."""

STATUS_BY_CODE = {
    "NOT_LOADED": 409,
    "UNKNOWN_ENTITY": 404,
    "UNPARSEABLE_CLAIM": 422,
    "BAD_REQUEST": 400,   # NOTE: not in HANDOFF.md section 5; added for malformed requests. Tell the team.
    "INTERNAL": 500,
}


class ModuleError(Exception):
    """Raised by modules/services; the server turns it into the uniform error body."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def status(self) -> int:
        return STATUS_BY_CODE.get(self.code, 500)

    def to_body(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}

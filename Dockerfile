# egrep.py as a container: the image runs egrep.py directly.
#
#   docker build -t egrep-py .
#   docker run --rm egrep-py "S(a|g|r)+on" 56667-0.txt            # book shipped in the image
#   docker run --rm -v "$PWD":/data egrep-py --stats "Babylon" /data/other.txt
#   echo "Sargon of Akkad" | docker run --rm -i egrep-py "S(a|g|r)+on"
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# egrep.py only needs the standard library.
COPY egrep.py syntax_tree.py alphabet.py nfa.py dfa.py kmp.py ./
COPY 56667-0.txt ./

RUN useradd --create-home --shell /usr/sbin/nologin egrep
USER egrep

ENTRYPOINT ["python", "/app/egrep.py"]

import flask
import csv
from flask import Flask, render_template, request
import difflib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import ast
import re
import urllib.request
import urllib.parse
import json
import os
import ssl

try:
    import certifi  # type: ignore
except Exception:
    certifi = None


POSTER_CACHE = {}
NO_POSTER_URL = "https://picsum.photos/seed/cine-fallback/500/750"
POSTER_DIR = os.path.join(os.path.dirname(__file__), "static", "posters")


def _urlopen(req, timeout=5):
    """
    urllib wrapper that works on Windows Python installs lacking CA roots.
    Prefers certifi bundle when available; otherwise falls back to unverified context.
    """
    try:
        if certifi is not None:
            ctx = ssl.create_default_context(cafile=certifi.where())
        else:
            ctx = ssl._create_unverified_context()
        return urllib.request.urlopen(req, timeout=timeout, context=ctx)
    except TypeError:
        # Older urlopen implementations may not accept context kwarg.
        return urllib.request.urlopen(req, timeout=timeout)


def _download_to_local(url, file_name):
    os.makedirs(POSTER_DIR, exist_ok=True)
    target_path = os.path.join(POSTER_DIR, file_name)
    if os.path.exists(target_path):
        # Replace old cached "no image available" placeholders.
        try:
            with open(target_path, "rb") as f:
                head = f.read(2048).lower()
            if b"no image available" not in head and b"wikimedia" not in head:
                return f"/static/posters/{file_name}"
        except Exception:
            return f"/static/posters/{file_name}"
        try:
            os.remove(target_path)
        except Exception:
            return f"/static/posters/{file_name}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with _urlopen(req, timeout=2.5) as response:
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type:
                return None
            data = response.read()
        if not data:
            return None
        with open(target_path, "wb") as f:
            f.write(data)
        return f"/static/posters/{file_name}"
    except Exception:
        return None


def _fallback_poster_local(file_name, seed_text):
    seed = urllib.parse.quote_plus((seed_text or "movie-poster")[:40])
    fallback_url = f"https://picsum.photos/seed/{seed}/500/750"
    return _download_to_local(fallback_url, file_name)


def _tmdb_search_poster(movie_title, release_date, file_name):
    try:
        query_parts = [movie_title]
        if release_date and isinstance(release_date, str) and len(release_date) >= 4:
            query_parts.append(release_date[:4])
        query = urllib.parse.quote_plus(" ".join([p for p in query_parts if p]))
        search_url = f"https://www.themoviedb.org/search?query={query}"
        req = urllib.request.Request(search_url, headers={"User-Agent": "Mozilla/5.0"})
        with _urlopen(req, timeout=3.5) as response:
            html = response.read().decode("utf-8", errors="ignore")
        poster_match = re.search(r"https://media\.themoviedb\.org/t/p/[^\"']+\.(?:jpg|png)", html)
        if poster_match:
            return _download_to_local(poster_match.group(0), file_name)
    except Exception:
        return None
    return None


def _omdb_poster(movie_title, release_date, file_name):
    year = ""
    if release_date and isinstance(release_date, str) and len(release_date) >= 4:
        year = release_date[:4]
    try:
        q = urllib.parse.quote_plus(movie_title)
        omdb_url = f"https://www.omdbapi.com/?t={q}&apikey=thewdb"
        if year.isdigit():
            omdb_url += f"&y={year}"
        req = urllib.request.Request(omdb_url, headers={"User-Agent": "Mozilla/5.0"})
        with _urlopen(req, timeout=2.8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        poster = payload.get("Poster")
        if poster and poster != "N/A":
            local_url = _download_to_local(poster, file_name)
            if local_url:
                return local_url
    except Exception:
        pass
    return None


def poster_image_url(movie_id, movie_title="", release_date=""):
    """Return real movie poster; always fall back to a real image."""
    cache_key = f"{movie_id}:{movie_title}".lower()
    if cache_key in POSTER_CACHE:
        return POSTER_CACHE[cache_key]

    try:
        tmdb_id = int(float(movie_id))
    except (TypeError, ValueError):
        tmdb_id = abs(hash(cache_key)) % 10_000_000

    try:
        req = urllib.request.Request(
            f"https://www.themoviedb.org/movie/{tmdb_id}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with _urlopen(req, timeout=0.9) as response:
            html = response.read().decode("utf-8", errors="ignore")
        match = re.search(r'<meta property="og:image" content="([^"]+)"', html)
        if match:
            poster_url = match.group(1)
            local_name = f"{tmdb_id}_v2.jpg"
            local_url = _download_to_local(poster_url, local_name)
            POSTER_CACHE[cache_key] = local_url or poster_url
            return POSTER_CACHE[cache_key]
    except Exception:
        pass

    local_name = f"{tmdb_id}_v2.jpg"

    # If TMDB page poster is unavailable, try TMDB search by title/year.
    if movie_title:
        local_url = _tmdb_search_poster(movie_title, release_date, local_name)
        if local_url:
            POSTER_CACHE[cache_key] = local_url
            return local_url

    # Reliable fallback: OMDb poster by title/year.
    if movie_title:
        local_url = _omdb_poster(movie_title, release_date, local_name)
        if local_url:
            POSTER_CACHE[cache_key] = local_url
            return local_url

    # Final fallback: always use a real image.
    local_url = _fallback_poster_local(local_name, movie_title or str(tmdb_id))
    if local_url:
        POSTER_CACHE[cache_key] = local_url
        return local_url

    POSTER_CACHE[cache_key] = NO_POSTER_URL
    return NO_POSTER_URL


def scrape_tmdb_top_rated(limit=18):
    """
    Web-scrape TMDB "Top Rated Movies" page (no API key).
    This source is significantly less likely to block requests than IMDb.
    """
    url = "https://www.themoviedb.org/movie/top-rated"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with _urlopen(req, timeout=10) as response:
        # Read enough HTML to cover the first page of cards.
        html = response.read(250_000).decode("utf-8", errors="ignore")

    # TMDB markup varies by locale and A/B tests. We therefore:
    # - grab movie IDs from hrefs
    # - infer title from a nearby title="" attribute when available
    # - infer release date and score from the nearby card HTML chunk
    link_re = re.compile(r'href="/movie/(?P<id>\d+)[^"]*"', flags=re.I)
    title_attr_re = re.compile(r'title="(?P<title>[^"]+)"', flags=re.I)
    date_re = re.compile(r'class="release_date"[^>]*>(?P<date>[^<]*)</span>', flags=re.I)
    score_re = re.compile(r'data-percent="(?P<percent>\d+)"', flags=re.I)
    poster_re = re.compile(r'(https://media\.themoviedb\.org/t/p/[^"\s]+)', flags=re.I)

    seen = set()
    out = []
    for m in link_re.finditer(html):
        mid = m.group("id")
        if not mid or mid in seen:
            continue
        seen.add(mid)

        # Look ahead within a small window for fields from the same card.
        chunk = html[m.start() : m.start() + 1500]
        t = title_attr_re.search(chunk)
        title = (t.group("title") if t else "").strip()
        d = date_re.search(chunk)
        date = (d.group("date") if d else "").strip()
        year = date[:4] if len(date) >= 4 else ""
        s = score_re.search(chunk)
        percent = s.group("percent") if s else None
        rating = round(int(percent) / 10.0, 1) if percent and percent.isdigit() else None

        p = poster_re.search(chunk)
        poster = p.group(1) if p else NO_POSTER_URL

        # If title is still missing, use a stable fallback.
        if not title:
            title = f"TMDB #{mid}"

        out.append(
            {
                "title": title,
                "year": year,
                "rating": rating,
                "url": f"https://www.themoviedb.org/movie/{mid}",
                "poster": poster,
            }
        )
        if len(out) >= max(0, int(limit)):
            break
    return out


app = flask.Flask(__name__, template_folder='templates')
df2 = pd.read_csv('tmdb.csv')
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])
all_titles = [df2['title'][i] for i in range(len(df2['title']))]
def extract_genres(genre_cell):
    if pd.isna(genre_cell):
        return []
    try:
        parsed = ast.literal_eval(genre_cell)
        if isinstance(parsed, list):
            genres = []
            for item in parsed:
                if isinstance(item, dict) and item.get('name'):
                    genres.append(str(item.get('name')).strip().title())
                elif isinstance(item, str) and item.strip():
                    genres.append(item.strip().title())
            return genres
    except (ValueError, SyntaxError):
        pass
    return []


df2['genre_list'] = df2['genres'].apply(extract_genres)
available_genres = sorted({genre for glist in df2['genre_list'] for genre in glist})


def get_recommendations(title, genre_filter='All'):
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]
    if genre_filter and genre_filter != 'All':
        filtered_scores = [s for s in sim_scores if genre_filter in df2['genre_list'].iloc[s[0]]]
        if len(filtered_scores) >= 10:
            sim_scores = filtered_scores
        elif filtered_scores:
            # Blend filtered and unfiltered to keep 10 results.
            chosen = list(filtered_scores)
            chosen_ids = {m[0] for m in chosen}
            for cand in sim_scores:
                if cand[0] not in chosen_ids:
                    chosen.append(cand)
                if len(chosen) >= 10:
                    break
            sim_scores = chosen
    sim_scores = sim_scores[:10]
    movie_indices = [i[0] for i in sim_scores]
    tit = df2['title'].iloc[movie_indices]
    dat = df2['release_date'].iloc[movie_indices]
    rating = df2['vote_average'].iloc[movie_indices]
    moviedetails=df2['overview'].iloc[movie_indices]
    movietypes=df2['keywords'].iloc[movie_indices]
    movieid=df2['id'].iloc[movie_indices]
    return_df = pd.DataFrame(columns=['Title','Year'])
    return_df['Title'] = tit
    return_df['Year'] = dat
    return_df['Ratings'] = rating
    return_df['Overview']=moviedetails
    return_df['Types']=movietypes
    return_df['ID']=movieid
    return return_df

def get_suggestions():
    data = pd.read_csv('tmdb.csv')
    return list(data['title'].str.capitalize())


def _to_int(value):
    try:
        if value is None:
            return None
        value = str(value).strip()
        if value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _to_float(value):
    try:
        if value is None:
            return None
        value = str(value).strip()
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def apply_result_filters(frame, min_rating=None, year_from=None, year_to=None, sort_mode="similar"):
    """Filter and sort a recommendation frame (Title/Year/Ratings/...)."""
    if frame is None or frame.empty:
        return frame

    f = frame.copy()
    f["__rating"] = pd.to_numeric(f.get("Ratings"), errors="coerce")
    f["__year"] = pd.to_datetime(f.get("Year"), errors="coerce").dt.year

    if min_rating is not None:
        f = f[f["__rating"].fillna(-1) >= float(min_rating)]

    if year_from is not None:
        f = f[f["__year"].fillna(-1) >= int(year_from)]

    if year_to is not None:
        f = f[f["__year"].fillna(10_000) <= int(year_to)]

    if sort_mode == "rating_desc":
        f = f.sort_values(by=["__rating", "__year"], ascending=[False, False], na_position="last")
    elif sort_mode == "rating_asc":
        f = f.sort_values(by=["__rating", "__year"], ascending=[True, False], na_position="last")
    elif sort_mode == "date_desc":
        f = f.sort_values(by=["__year", "__rating"], ascending=[False, False], na_position="last")
    elif sort_mode == "date_asc":
        f = f.sort_values(by=["__year", "__rating"], ascending=[True, False], na_position="last")

    return f.drop(columns=["__rating", "__year"], errors="ignore")


def build_payload_from_frame(frame):
    names = []
    dates = []
    ratings = []
    overview = []
    types = []
    mid = []
    posters = []
    for _, row in frame.iterrows():
        names.append(row['title'])
        dates.append(row['release_date'])
        ratings.append(row['vote_average'])
        overview.append(row['overview'])
        types.append(", ".join(row['genre_list'][:2]) if row['genre_list'] else "General")
        mid.append(row['id'])
        posters.append(poster_image_url(row['id'], row['title'], row['release_date']))
    return names, dates, ratings, overview, types, mid, posters


def original_movie_payload(movie_title):
    row = df2[df2['title'] == movie_title]
    if row.empty:
        return {
            'original_movie_name': movie_title,
            'original_movie_date': '',
            'original_movie_poster': NO_POSTER_URL,
        }
    first = row.iloc[0]
    return {
        'original_movie_name': first['title'],
        'original_movie_date': first['release_date'],
        'original_movie_poster': poster_image_url(first['id'], first['title'], first['release_date']),
    }

app = Flask(__name__)
@app.route("/")
@app.route("/index")
def index():
    NewMovies=[]
    with open('movieR.csv','r') as csvfile:
        readCSV = csv.reader(csvfile)
        NewMovies.append(random.choice(list(readCSV)))
    m_name = NewMovies[0][0]
    m_name = m_name.title()
    
    with open('movieR.csv', 'a',newline='') as csv_file:
        fieldnames = ['Movie']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({'Movie': m_name})
        result_final = get_recommendations(m_name)
        names = []
        dates = []
        ratings = []
        overview=[]
        types=[]
        mid=[]
        posters = []
        for i in range(len(result_final)):
            names.append(result_final.iloc[i]["Title"])
            dates.append(result_final.iloc[i]["Year"])
            ratings.append(result_final.iloc[i]["Ratings"])
            overview.append(result_final.iloc[i]["Overview"])
            types.append(result_final.iloc[i]["Types"])
            mid.append(result_final.iloc[i]["ID"])
            posters.append(poster_image_url(result_final.iloc[i]["ID"], result_final.iloc[i]["Title"], result_final.iloc[i]["Year"]))
    suggestions = get_suggestions()

    return render_template(
        'index.html',
        suggestions=suggestions,
        genres=available_genres,
        **original_movie_payload(m_name),
        movie_type=types[5:],
        movieid=mid,
        movie_posters=posters,
        movie_overview=overview,
        movie_names=names,
        movie_date=dates,
        movie_ratings=ratings,
        search_name=m_name,
    )


@app.route("/genres")
def genre_page():
    selected_genre = flask.request.args.get('genre', '')
    if not selected_genre or selected_genre not in available_genres:
        selected_genre = available_genres[0] if available_genres else ''

    filtered = df2[df2['genre_list'].apply(lambda g: selected_genre in g)] if selected_genre else df2.head(0)
    filtered = filtered.sort_values(by=['vote_average', 'popularity'], ascending=False).head(18)
    names, dates, ratings, overview, types, mid, posters = build_payload_from_frame(filtered)

    return render_template(
        'genres.html',
        genres=available_genres,
        selected_genre=selected_genre,
        movie_type=types,
        movieid=mid,
        movie_posters=posters,
        movie_overview=overview,
        movie_names=names,
        movie_date=dates,
        movie_ratings=ratings,
        search_name=selected_genre,
    )


@app.route("/scrape")
def scrape_page():
    try:
        movies = scrape_tmdb_top_rated(limit=18)
    except Exception:
        movies = []
    return render_template("scrape.html", movies=movies)

# Set up the main route
@app.route('/positive', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return flask.render_template(
            'index.html',
            suggestions=get_suggestions(),
            genres=available_genres,
            selected_sort='similar',
            min_rating='',
            year_from='',
            year_to='',
        )

    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name']
        sort_mode = flask.request.form.get('sort', 'similar')
        min_rating = _to_float(flask.request.form.get('min_rating'))
        year_from = _to_int(flask.request.form.get('year_from'))
        year_to = _to_int(flask.request.form.get('year_to'))

        m_name = m_name.title()
        if m_name not in all_titles:
            return(flask.render_template('negative.html',name=m_name))
        else:
            with open('movieR.csv', 'a',newline='') as csv_file:
                fieldnames = ['Movie']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'Movie': m_name})
            result_final = get_recommendations(m_name, 'All')
            result_final = apply_result_filters(
                result_final,
                min_rating=min_rating,
                year_from=year_from,
                year_to=year_to,
                sort_mode=sort_mode,
            )
            names = []
            dates = []
            ratings = []
            overview=[]
            types=[]
            mid=[]
            posters = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i]["Title"])
                dates.append(result_final.iloc[i]["Year"])
                ratings.append(result_final.iloc[i]["Ratings"])
                overview.append(result_final.iloc[i]["Overview"])
                types.append(result_final.iloc[i]["Types"])
                mid.append(result_final.iloc[i]["ID"])
                posters.append(poster_image_url(result_final.iloc[i]["ID"], result_final.iloc[i]["Title"], result_final.iloc[i]["Year"]))

            return flask.render_template(
                'positive.html',
                genres=available_genres,
                selected_sort=sort_mode,
                min_rating=min_rating if min_rating is not None else '',
                year_from=year_from if year_from is not None else '',
                year_to=year_to if year_to is not None else '',
                **original_movie_payload(m_name),
                movie_type=types[5:],
                movieid=mid,
                movie_posters=posters,
                movie_overview=overview,
                movie_names=names,
                movie_date=dates,
                movie_ratings=ratings,
                search_name=m_name,
            )

if __name__ == '__main__':
    app.run()

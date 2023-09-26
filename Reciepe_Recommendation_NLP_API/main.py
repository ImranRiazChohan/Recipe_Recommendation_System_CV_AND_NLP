import rec_sys
import uvicorn
from fastapi import FastAPI,Query

app = FastAPI()

@app.get('/')
async def index():
    return {'mesaage':'Recipe_Recommendation System'}

@app.get("/recipe", response_model=dict)
def recommend_recipe(ingredients: str = Query(..., title="Ingredients", description="Comma-separated list of ingredients")):
    recipe = rec_sys.RecSys(ingredients)
    
    response = []
    for index, row in recipe.iterrows():
        response.append({
            'recipe': str(row['recipe']),
            'ingredients': str(row['ingredients']),
            'Cuisine' :str(row['Cuisine']),
            'Course' :str(row['Course']),
            'Diet': str(row['Diet'])
        })
    
    return {"Recipe":response}
if __name__ == "__main__":
    uvicorn.run(app,port=8000)

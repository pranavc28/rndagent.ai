Quick demo of an RND agent that answers questions about a biodegradable policy :)

Prerequisites to download and run:
- OpenAI API key
- Python 3.10+

Steps to run:
- Clone the repo
- Create a python virtual environment
- Create a python openai api key
- Run `export OPENAI_API_KEY=<your-api-key>` on your terminal
- Install dependencies above
- Run `python rnd_agent.py`

You can add more files to the vector store by adding them to the `files` folder, changing the tmpl or txt file and running `python rnd_agent.py` again.

Sample response:

```
Response: Transitioning to biodegradable packaging materials, specifically replacing PET plastics with biodegradable PLA, in a company with 1000 employees and 10000 products can be achieved through the following 5-step plan:

1. **Internal Awareness and Training**: Start by educating all employees about the benefits of transitioning to biodegradable packaging materials and the importance of sustainability initiatives. Conduct training sessions to raise awareness about the negative environmental impacts of PET plastics and the advantages of PLA in terms of biodegradability and lower carbon footprint. Engage employees from all departments, including production, procurement, and waste management, to ensure widespread understanding and cooperation[0].

2. **Material Testing and Selection**: Establish a dedicated R&D team to conduct thorough tests on PLA to assess its durability, shelf life, and functional characteristics in comparison to PET plastics. Ensure that PLA meets the required standards for strength, permeability, and resistance to environmental factors. Consider factors like transport efficiency and energy consumption in the production process. Identify and collaborate with certified suppliers of biodegradable PLA who adhere to international standards to ensure the quality and sustainability of the materials[0].

3. **Supplier Transition and Procurement Strategy**: Work closely with the Procurement department to transition from PET plastic suppliers to verified suppliers of biodegradable PLA. Ensure that the transition aligns with the company's sustainability goals and that the selected suppliers comply with essential certifications such as ISO 14855 or ASTM D6400 for biodegradability. Negotiate contracts that prioritize the procurement of PLA from sustainable and environmentally conscious sources[0].

4. **Packaging Design and Implementation**: Collaborate with the packaging design and production teams to integrate biodegradable PLA materials into the packaging of all 10000 products. Ensure that the packaging maintains the necessary functionality and protection provided by PET plastics while incorporating the sustainable benefits of PLA. Monitor the production process to guarantee that the transition to PLA packaging remains cost-effective and does not compromise on quality or performance[0].

5. **Waste Management and End-of-Life Considerations**: Develop a comprehensive waste management strategy in coordination with the Waste Management team to facilitate the proper composting and recycling of PLA packaging materials. Establish partnerships with local composting facilities and authorities to ensure the effective disposal of biodegradable materials. Implement initiatives to encourage consumers to dispose of PLA packaging appropriately to maximize composting and minimize environmental impact[0].

By following this 5-step plan emphasizing internal awareness, material testing, supplier transition, packaging design, and waste management, the company can successfully transition to biodegradable PLA packaging materials, thereby significantly reducing its environmental footprint and contributing to a more sustainable future.

Citations:
[0] ShenJ_2011.pdf
[1] ShenJ_2011.pdf
[2] ShenJ_2011.pdf
[3] ShenJ_2011.pdf
[4] ShenJ_2011.pdf
```

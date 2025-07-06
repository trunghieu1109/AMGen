async def forward_9(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Subtask 1: Identify the Shakespeare tragedy that includes the fictional character Benvolio and determine if Benvolio slays a character in it."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, identifying tragedy with Benvolio and slaying info, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    cot_sc_instruction2 = "Subtask 2: List possible protagonists of the identified Shakespeare tragedy from Subtask 1."
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc={
            'instruction': cot_sc_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
        },
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, listing protagonists, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    protagonists = results2['answer'].content if hasattr(results2['answer'], 'content') else results2['answer']
    if isinstance(protagonists, str):
        protagonists_list = [p.strip() for p in protagonists.split(',') if p.strip()]
    else:
        protagonists_list = protagonists
    protagonist_love_marriage_results = []
    for idx, protagonist in enumerate(protagonists_list, start=3):
        cot_instruction3 = f"Subtask {idx}: Check if protagonist {protagonist} secretly loves and marries a member of a rival house in the identified tragedy."
        cot_agent_desc3 = {
            'instruction': cot_instruction3,
            'input': [taskInfo, protagonist, results1['answer'].content],
            'temperature': 0.0,
            'context': ["user query", "protagonist info", "tragedy info"]
        }
        results3 = await self.cot(
            subtask_id=f"subtask_{idx}",
            cot_agent_desc=cot_agent_desc3
        )
        agents.append(f"CoT agent {results3['cot_agent'].id}, checking love and marriage for {protagonist}, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
        sub_tasks.append(f"Subtask {idx} output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
        logs.append(results3['subtask_desc'])
        protagonist_love_marriage_results.append((protagonist, results3['answer'].content))
    aggregate_instruction4 = "Subtask 4: Consolidate the candidate protagonists and their relationships to identify the correct character matching all criteria from previous subtasks."
    aggregate_desc4 = {
        'instruction': aggregate_instruction4,
        'input': [taskInfo, results1['answer'].content, results2['answer'].content] + [res[1] for res in protagonist_love_marriage_results],
        'temperature': 0.0,
        'context': ["user query", "previous subtask answers"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc4
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, consolidating protagonist and relationship info, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    review_instruction5 = "Subtask 5: Validate the consolidated output to ensure it correctly answers the query about the protagonist who secretly loves and marries a rival house member."
    review_desc5 = {
        'instruction': review_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "consolidated output"]
    }
    results5 = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc5
    )
    agents.append(f"Review agent {results5['review_agent'].id}, validating consolidated output, feedback: {results5['thinking'].content}; correct: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: feedback - {results5['thinking'].content}; correct - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs

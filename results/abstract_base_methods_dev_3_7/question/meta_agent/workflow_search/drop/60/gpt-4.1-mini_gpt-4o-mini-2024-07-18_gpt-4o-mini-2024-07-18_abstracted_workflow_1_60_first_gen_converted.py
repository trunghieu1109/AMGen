async def forward_60(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Subtask 1: Extract all touchdown runs mentioned in the passage along with their yardages from the given text."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting touchdown runs, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    td_runs_text = results1['answer'].content
    td_runs_list = []
    import re
    pattern = r'(\d+)[- ]?yard(?:s)? TD run'
    matches = re.findall(pattern, td_runs_text)
    for m in matches:
        try:
            yard = int(m)
            td_runs_list.append(yard)
        except:
            continue
    filtered_runs = []
    for idx, yardage in enumerate(td_runs_list, start=1):
        cot_instruction2 = f"Subtask 2: Determine if the touchdown run of {yardage} yards is under 10 yards."
        cot_agent_desc2 = {
            'instruction': cot_instruction2,
            'input': [taskInfo, str(yardage)],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results2 = await self.cot(
            subtask_id=f"subtask_2_{idx}",
            cot_agent_desc=cot_agent_desc2
        )
        agents.append(f"CoT agent {results2['cot_agent'].id}, evaluating run yardage {yardage}, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
        sub_tasks.append(f"Subtask 2_{idx} output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])
        answer_lower = results2['answer'].content.lower()
        if 'under 10' in answer_lower or 'less than 10' in answer_lower or 'yes' in answer_lower:
            filtered_runs.append(yardage)
    debate_instruction3 = f"Subtask 3: Count the number of touchdown runs with yardage under 10 yards based on the filtered results: {filtered_runs}."
    final_decision_instruction3 = "Subtask 3: Make final decision on the count of touchdown runs under 10 yards."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", f"filtered runs: {filtered_runs}"],
        'input': [taskInfo, str(filtered_runs)],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc3 = {
        'instruction': final_decision_instruction3,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        final_decision_desc=final_decision_desc3,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results3['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, counting filtered runs, thinking: {results3['list_thinking'][round][idx].content}; answer: {results3['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, counting filtered runs, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'], sub_tasks, agents)
    return final_answer, logs
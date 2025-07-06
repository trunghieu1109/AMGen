async def forward_10(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Subtask 1: Identify and list all scoring events in the passage that mention field goals, including the kicker's name and the yardage of each field goal." 
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, identifying all field goal scoring events, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    scoring_events_text = results1['answer'].content
    import re
    pattern = r"(\d+)-yard field goal by ([^.,]+)"
    matches = re.findall(pattern, scoring_events_text)
    total_yardages = []
    for idx, (yardage, kicker) in enumerate(matches, start=1):
        cot_instruction2 = f"Subtask 2: For scoring event {idx}, determine if the field goal was made by Nate Kaeding. Kicker: {kicker}, Yardage: {yardage}."
        cot_agent_desc2 = {
            'instruction': cot_instruction2,
            'input': [taskInfo, scoring_events_text],
            'temperature': 0.0,
            'context': ["user query", "field goal events"]
        }
        results2 = await self.cot(
            subtask_id=f"subtask_2_{idx}",
            cot_agent_desc=cot_agent_desc2
        )
        agents.append(f"CoT agent {results2['cot_agent'].id}, checking if field goal {idx} was by Nate Kaeding, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
        sub_tasks.append(f"Subtask 2_{idx} output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])
        if 'yes' in results2['answer'].content.lower():
            cot_instruction3 = f"Subtask 3: Extract the yardage of the field goal made by Nate Kaeding for scoring event {idx}."
            cot_agent_desc3 = {
                'instruction': cot_instruction3,
                'input': [taskInfo, scoring_events_text, results2['answer'].content],
                'temperature': 0.0,
                'context': ["user query", "field goal events", "Nate Kaeding field goals"]
            }
            results3 = await self.cot(
                subtask_id=f"subtask_3_{idx}",
                cot_agent_desc=cot_agent_desc3
            )
            agents.append(f"CoT agent {results3['cot_agent'].id}, extracting yardage for Nate Kaeding field goal {idx}, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
            sub_tasks.append(f"Subtask 3_{idx} output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
            logs.append(results3['subtask_desc'])
            try:
                yard = int(re.search(r'\d+', results3['answer'].content).group())
                total_yardages.append(yard)
            except:
                pass
    total_yards = sum(total_yardages)
    debate_instruction_4 = f"Subtask 4: Sum all extracted yardages of Nate Kaeding's field goals to find the total yards made. Yardages: {total_yardages}."
    final_decision_instruction_4 = "Subtask 4: Make final decision on the total yards of field goals made by Nate Kaeding."
    debate_desc4 = {
        'instruction': debate_instruction_4,
        'context': ["user query", f"yardages: {total_yardages}"],
        'input': [taskInfo, str(total_yardages)],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc4 = {
        'instruction': final_decision_instruction_4,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        final_decision_desc=final_decision_desc4,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, summing yardages and calculating total, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, calculating total yards, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs

async def forward_77(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Subtask 1: Identify and list all scoring events mentioned in the passage, including field goals, touchdowns, safeties, and other scoring plays from the given passage." 
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, identifying scoring events, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    scoring_events = results1['answer'].content
    successful_field_goals = []
    for idx, event in enumerate(scoring_events.split('\n')):
        cot_instruction2 = f"Subtask 2: For the scoring event '{event}', determine if it is a field goal attempt." 
        cot_agent_desc2 = {
            'instruction': cot_instruction2,
            'input': [taskInfo, event],
            'temperature': 0.0,
            'context': ["user query", "scoring event analysis"]
        }
        results2 = await self.cot(
            subtask_id=f"subtask_2_{idx+1}",
            cot_agent_desc=cot_agent_desc2
        )
        agents.append(f"CoT agent {results2['cot_agent'].id}, checking if field goal, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
        sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])
        is_field_goal = results2['answer'].content.strip().lower() == 'yes'
        if is_field_goal:
            cot_instruction3 = f"Subtask 3: For the field goal event '{event}', determine whether it was successful." 
            cot_agent_desc3 = {
                'instruction': cot_instruction3,
                'input': [taskInfo, event],
                'temperature': 0.0,
                'context': ["user query", "field goal success analysis"]
            }
            results3 = await self.cot(
                subtask_id=f"subtask_3_{idx+1}",
                cot_agent_desc=cot_agent_desc3
            )
            agents.append(f"CoT agent {results3['cot_agent'].id}, checking field goal success, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
            sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
            logs.append(results3['subtask_desc'])
            if results3['answer'].content.strip().lower() == 'yes':
                successful_field_goals.append(event)
    debate_instruction4 = f"Subtask 4: Count the total number of successful field goals from the identified successful field goal events." 
    final_decision_instruction4 = "Subtask 4: Make final decision on the total count of successful field goals." 
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", "successful field goals list"],
        'input': [taskInfo, '\n'.join(successful_field_goals)],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc4 = {
        'instruction': final_decision_instruction4,
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
            agents.append(f"Debate agent {agent.id}, round {round}, counting successful field goals, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, counting successful field goals, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs

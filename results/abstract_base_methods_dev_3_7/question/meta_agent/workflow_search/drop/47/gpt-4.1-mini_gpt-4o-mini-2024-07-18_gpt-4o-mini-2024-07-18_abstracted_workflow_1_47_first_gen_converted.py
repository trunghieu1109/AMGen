async def forward_47(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Parse the passage to identify all scoring plays and list them in chronological order with their details (type of score, player, yardage)."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, parsing passage for scoring plays, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    scoring_plays = results1['answer'].content

    touchdown_count = 0
    second_touchdown_yardage = None

    for idx, play in enumerate(scoring_plays):
        cot_instruction2 = f"Subtask 2: Check if scoring play #{idx+1} is a touchdown based on its details."
        cot_agent_desc2 = {
            'instruction': cot_instruction2,
            'input': [play],
            'temperature': 0.0,
            'context': ["user query", "scoring play details"]
        }
        results2 = await self.cot(
            subtask_id=f"subtask_2_{idx+1}",
            cot_agent_desc=cot_agent_desc2
        )
        agents.append(f"CoT agent {results2['cot_agent'].id}, checking touchdown status for play #{idx+1}, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
        sub_tasks.append(f"Subtask 2_{idx+1} output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])

        is_touchdown = results2['answer'].content.strip().lower() == 'yes'

        if is_touchdown:
            touchdown_count += 1
            cot_instruction3 = f"Subtask 3: Since play #{idx+1} is a touchdown, increment touchdown count to {touchdown_count}."
            cot_agent_desc3 = {
                'instruction': cot_instruction3,
                'input': [touchdown_count],
                'temperature': 0.0,
                'context': ["user query", "touchdown count"]
            }
            results3 = await self.cot(
                subtask_id=f"subtask_3_{idx+1}",
                cot_agent_desc=cot_agent_desc3
            )
            agents.append(f"CoT agent {results3['cot_agent'].id}, incrementing touchdown count, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
            sub_tasks.append(f"Subtask 3_{idx+1} output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
            logs.append(results3['subtask_desc'])

            cot_instruction4 = f"Subtask 4: Check if touchdown count {touchdown_count} equals 2 (second touchdown)."
            cot_agent_desc4 = {
                'instruction': cot_instruction4,
                'input': [touchdown_count],
                'temperature': 0.0,
                'context': ["user query", "touchdown count check"]
            }
            results4 = await self.cot(
                subtask_id=f"subtask_4_{idx+1}",
                cot_agent_desc=cot_agent_desc4
            )
            agents.append(f"CoT agent {results4['cot_agent'].id}, checking if second touchdown, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
            sub_tasks.append(f"Subtask 4_{idx+1} output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
            logs.append(results4['subtask_desc'])

            is_second_touchdown = results4['answer'].content.strip().lower() == 'yes'

            if is_second_touchdown:
                cot_instruction5 = f"Subtask 5: Extract the yardage of the second touchdown from play #{idx+1} details."
                cot_agent_desc5 = {
                    'instruction': cot_instruction5,
                    'input': [play],
                    'temperature': 0.0,
                    'context': ["user query", "second touchdown yardage extraction"]
                }
                results5 = await self.cot(
                    subtask_id=f"subtask_5_{idx+1}",
                    cot_agent_desc=cot_agent_desc5
                )
                agents.append(f"CoT agent {results5['cot_agent'].id}, extracting yardage for second touchdown, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
                sub_tasks.append(f"Subtask 5_{idx+1} output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
                logs.append(results5['subtask_desc'])

                second_touchdown_yardage = results5['answer'].content.strip()
                break

    if second_touchdown_yardage is None:
        second_touchdown_yardage = "Not found"

    cot_instruction6 = f"Subtask 6: Store the yardage '{second_touchdown_yardage}' of the second touchdown for final output."
    cot_agent_desc6 = {
        'instruction': cot_instruction6,
        'input': [second_touchdown_yardage],
        'temperature': 0.0,
        'context': ["user query", "final yardage storage"]
    }
    results6 = await self.cot(
        subtask_id="subtask_6",
        cot_agent_desc=cot_agent_desc6
    )
    agents.append(f"CoT agent {results6['cot_agent'].id}, storing second touchdown yardage, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    debate_instruction7 = f"Subtask 7: Output the yardage of the second touchdown as the final answer."
    debate_desc7 = {
        'instruction': debate_instruction7,
        'context': ["user query", "second touchdown yardage"],
        'input': [second_touchdown_yardage],
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    final_decision_desc7 = {
        'instruction': "Subtask 7: Make final decision on the yardage of the second touchdown.",
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results7 = await self.debate(
        subtask_id="subtask_7",
        debate_desc=debate_desc7,
        final_decision_desc=final_decision_desc7,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results7['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, finalizing output, thinking: {results7['list_thinking'][round][idx].content}; answer: {results7['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, finalizing yardage output, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
    sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    final_answer = await self.make_final_answer(results7['thinking'], results7['answer'], sub_tasks, agents)
    return final_answer, logs

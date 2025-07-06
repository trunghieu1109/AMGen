async def forward_66(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Identify the year of Ó Snodaigh's first FAI Cup win from the passage."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, identifying first FAI Cup win year, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction2 = "Subtask 2: Identify the year of Ó Snodaigh's second FAI Cup win from the passage, if any, based on the output from Subtask 1."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, identifying second FAI Cup win year, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    second_win_year_text = results2['answer'].content.strip().lower()
    has_second_win = second_win_year_text not in ['', 'none', 'no', 'not mentioned', 'no second win']

    if has_second_win:
        cot_instruction3 = "Subtask 3: Calculate the number of years between the first and second FAI Cup wins based on the identified years."
        cot_agent_desc3 = {
            'instruction': cot_instruction3,
            'input': [results1['answer'], results2['answer']],
            'temperature': 0.0,
            'context': ["user query", "answer of subtask 1", "answer of subtask 2"]
        }
        results3 = await self.cot(
            subtask_id="subtask_3",
            cot_agent_desc=cot_agent_desc3
        )
        agents.append(f"CoT agent {results3['cot_agent'].id}, calculating year difference, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
        sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
        logs.append(results3['subtask_desc'])

        debate_instruction4 = "Subtask 4: Generate the final answer stating the number of years between the first and second FAI Cup wins."
        final_decision_instruction4 = "Subtask 4: Make final decision on the number of years between the first and second FAI Cup wins."
        debate_desc4 = {
            'instruction': debate_instruction4,
            'context': ["user query", "thinking of subtask 3", "answer of subtask 3"],
            'input': [taskInfo, results3['thinking'], results3['answer']],
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
                agents.append(f"Debate agent {agent.id}, round {round}, generating final answer, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
        agents.append(f"Final Decision agent, finalizing answer, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
        sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
        logs.append(results4['subtask_desc'])

        final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
        return final_answer, logs

    else:
        debate_instruction5 = "Subtask 5: Generate the final answer stating that there is no second FAI Cup win mentioned in the passage."
        final_decision_instruction5 = "Subtask 5: Make final decision on the absence of a second FAI Cup win."
        debate_desc5 = {
            'instruction': debate_instruction5,
            'context': ["user query", "thinking of subtask 2", "answer of subtask 2"],
            'input': [taskInfo, results2['thinking'], results2['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.5
        }
        final_decision_desc5 = {
            'instruction': final_decision_instruction5,
            'output': ["thinking", "answer"],
            'temperature': 0.0
        }
        results5 = await self.debate(
            subtask_id="subtask_5",
            debate_desc=debate_desc5,
            final_decision_desc=final_decision_desc5,
            n_repeat=self.max_round
        )
        for round in range(self.max_round):
            for idx, agent in enumerate(results5['debate_agent']):
                agents.append(f"Debate agent {agent.id}, round {round}, generating final answer for no second win, thinking: {results5['list_thinking'][round][idx].content}; answer: {results5['list_answer'][round][idx].content}")
        agents.append(f"Final Decision agent, finalizing answer for no second win, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
        sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
        logs.append(results5['subtask_desc'])

        final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
        return final_answer, logs

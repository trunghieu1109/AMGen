async def forward_82(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Identify all passing touchdowns thrown by Donovan McNabb in the passage."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, identifying passing touchdowns by Donovan McNabb, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction2 = "Subtask 2: Extract the yardage for each passing touchdown identified for Donovan McNabb from Subtask 1."
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
    agents.append(f"CoT agent {results2['cot_agent'].id}, extracting yardages for passing touchdowns, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    yardages_text = results2['answer'].content
    yardages = []
    import re
    matches = re.findall(r'\b(\d+)\b', yardages_text)
    for m in matches:
        yardages.append(int(m))

    total_yards = sum(yardages)

    thinking3 = f"Summed the yardages {yardages} to get total passing touchdown yards: {total_yards}."
    answer3 = f"{total_yards}"
    sub_tasks.append(f"Subtask 3 output: thinking - {thinking3}; answer - {answer3}")
    agents.append(f"Aggregate agent, summing yardages {yardages} to total {total_yards}")

    cot_reflect_instruction4 = "Subtask 4: Review the summed total to ensure correctness and consistency with the passage details."
    critic_instruction4 = "Please review the correctness and consistency of the total passing touchdown yards calculated."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], thinking3, answer3],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    critic_desc4 = {
        'instruction': critic_instruction4,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results4 = await self.reflexion(
        subtask_id="subtask_4",
        cot_reflect_desc=cot_reflect_desc4,
        critic_desc=critic_desc4,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, reviewing total yards, thinking: {results4['list_thinking'][0].content}; answer: {results4['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results4['critic_agent'].id}, feedback round {i}, thinking: {results4['list_feedback'][i].content}; answer: {results4['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining review round {i}, thinking: {results4['list_thinking'][i + 1].content}; answer: {results4['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    debate_instruction5 = "Subtask 5: Generate the final concise answer stating the total yards Donovan McNabb accumulated on passing touchdowns."
    final_decision_instruction5 = "Subtask 5: Make final decision on the total passing touchdown yards of Donovan McNabb."
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"],
        'input': [taskInfo, results4['thinking'], results4['answer']],
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
            agents.append(f"Debate agent {agent.id}, round {round}, generating final answer, thinking: {results5['list_thinking'][round][idx].content}; answer: {results5['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs

async def forward_8(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Start sequential flow
    
    # Start loop flow for iterative refinement
    for i in range(self.max_round):
        
        # Subtask 1: Iteratively evaluate and improve the clarity, consistency, and completeness of the reasoning and calculation for determining the percentage of people not Native American in 2000 using Reflexion
        cot_reflect_instruction = "Subtask 1: Iteratively evaluate and improve the clarity, consistency, and completeness of the reasoning and calculation for determining the percentage of people not Native American in 2000."
        critic_instruction = "Please review the reasoning and calculation for determining the percentage of people not Native American in 2000 and provide feedback on its limitations."
        cot_reflect_desc = {
            'instruction': cot_reflect_instruction,
            'input': [taskInfo],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query"]
        }
        critic_desc = {
            'instruction': critic_instruction,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results_reflexion = await self.reflexion(
            subtask_id="subtask_1",
            cot_reflect_desc=cot_reflect_desc,
            critic_desc=critic_desc,
            n_repeat=1
        )
        agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, thinking: {results_reflexion['list_thinking'][0].content}; answer: {results_reflexion['list_answer'][0].content}")
        for idx in range(min(1, len(results_reflexion['list_feedback']))):
            agents.append(f"Critic agent {results_reflexion['critic_agent'].id}, feedback: {results_reflexion['list_feedback'][idx].content}; correction: {results_reflexion['list_correct'][idx].content}")
        sub_tasks.append(f"Subtask 1 output: thinking - {results_reflexion['thinking'].content}; answer - {results_reflexion['answer'].content}")
        logs.append(results_reflexion['subtask_desc'])
        
        # Subtask 2: Decompose the census data from 2000 into logical steps to calculate the percentage of people who were not Native American using CoT and AnswerGenerate
        cot_instruction = "Subtask 2: Decompose the census data from 2000 into logical steps to calculate the percentage of people who were not Native American."
        cot_agent_desc = {
            'instruction': cot_instruction,
            'input': [taskInfo, results_reflexion['thinking'], results_reflexion['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results_cot = await self.answer_generate(
            subtask_id="subtask_2",
            cot_agent_desc=cot_agent_desc
        )
        agents.append(f"CoT agent {results_cot['cot_agent'].id}, thinking: {results_cot['thinking'].content}; answer: {results_cot['answer'].content}")
        sub_tasks.append(f"Subtask 2 output: thinking - {results_cot['thinking'].content}; answer - {results_cot['answer'].content}")
        logs.append(results_cot['subtask_desc'])
    
    # End loop flow
    
    # End sequential flow
    
    final_answer = await self.make_final_answer(results_cot['thinking'], results_cot['answer'], sub_tasks, agents)
    return final_answer, logs
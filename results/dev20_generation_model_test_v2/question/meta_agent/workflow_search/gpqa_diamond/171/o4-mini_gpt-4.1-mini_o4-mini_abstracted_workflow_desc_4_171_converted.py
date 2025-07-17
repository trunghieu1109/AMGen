async def forward_171(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 0: SC-CoT to derive N1/N2 = 2 using Boltzmann formula
    N = self.max_sc
    cot_agents0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings0 = []
    possible_answers0 = []
    subtask_desc0 = {
        'subtask_id': 'subtask_0',
        'instruction': 'Sub-task 0: Write the ratio of excited-level populations N1/N2 using the Boltzmann formula in LTE, showing it equals 2.',
        'context': ['user query'],
        'agent_collaboration': 'SC_CoT'
    }
    for i in range(N):
        thinking0_i, answer0_i = await cot_agents0[i]([taskInfo], subtask_desc0['instruction'], is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents0[i].id}, deriving population ratio, thinking: {thinking0_i.content}; answer: {answer0_i.content}")
        possible_thinkings0.append(thinking0_i)
        possible_answers0.append(answer0_i)
    final_agent0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_agent0([taskInfo] + possible_thinkings0 + possible_answers0, 'Sub-task 0: Synthesize and choose the most consistent answer for the population ratio.', is_sub_task=True)
    agents.append(f"Final Decision Agent, synthesizing population ratio, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {'thinking': thinking0, 'answer': answer0}
    logs.append(subtask_desc0)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 1: SC-CoT to introduce ΔE and k_B into the ratio
    cot_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {
        'subtask_id': 'subtask_1',
        'instruction': 'Sub-task 1: Introduce ΔE and k_B into the ratio to express 2 = exp[-ΔE/(k_B T1) + ΔE/(k_B T2)] and simplify the exponent.',
        'context': ['user query', thinking0, answer0],
        'agent_collaboration': 'SC_CoT'
    }
    for i in range(N):
        thinking1_i, answer1_i = await cot_agents1[i]([taskInfo, thinking0, answer0], subtask_desc1['instruction'], is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, introducing ΔE and k_B, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_agent1([taskInfo, thinking0, answer0] + possible_thinkings1 + possible_answers1, 'Sub-task 1: Synthesize and choose the most consistent exponent simplification.', is_sub_task=True)
    agents.append(f"Final Decision Agent, synthesizing exponent form, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 2: Debate to isolate ln(2) relation
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    subtask_desc2 = {
        'subtask_id': 'subtask_2',
        'instruction': 'Sub-task 2: Rearrange the exponent to isolate ln(2) = (ΔE/k_B)*(1/T2 - 1/T1), yielding the target temperature relation.',
        'context': ['user query', thinking1, answer1],
        'agent_collaboration': 'Debate'
    }
    debate_agents2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking2 = [[] for _ in range(N_max)]
    all_answer2 = [[] for _ in range(N_max)]
    for r in range(N_max):
        for i, agent in enumerate(debate_agents2):
            if r == 0:
                inputs = [taskInfo, thinking1, answer1]
            else:
                inputs = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
            thinking2_i, answer2_i = await agent(inputs, subtask_desc2['instruction'] + debate_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
            all_thinking2[r].append(thinking2_i)
            all_answer2[r].append(answer2_i)
    final_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_agent2([taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1], 'Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.', is_sub_task=True)
    agents.append(f"Final Decision Agent, finalizing ln(2) relation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 3: Debate to match derived relation with choices
    subtask_desc3 = {
        'subtask_id': 'subtask_3',
        'instruction': 'Sub-task 3: Compare the derived relation ln(2) = (ΔE/k_B)*(1/T2 - 1/T1) with the four choices and identify which matches.',
        'context': ['user query', thinking2, answer2],
        'agent_collaboration': 'Debate'
    }
    debate_agents3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(N_max)]
    all_answer3 = [[] for _ in range(N_max)]
    for r in range(N_max):
        for i, agent in enumerate(debate_agents3):
            if r == 0:
                inputs = [taskInfo, thinking2, answer2]
            else:
                inputs = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
            thinking3_i, answer3_i = await agent(inputs, subtask_desc3['instruction'] + debate_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
            all_thinking3[r].append(thinking3_i)
            all_answer3[r].append(answer3_i)
    final_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_agent3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], 'Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.', is_sub_task=True)
    agents.append(f"Final Decision Agent, selecting matching choice, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
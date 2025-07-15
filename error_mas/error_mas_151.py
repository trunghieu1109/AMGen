async def forward_151(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 0.1: Extract and classify key elements
    cot_sc_instruction = "Sub-task 0.1: Extract and classify the biological system, observation, analytical aim, methodology, and candidate complexes from the query."
    N0 = self.max_sc
    cot_agents0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N0)]
    possible_thinkings0 = []
    possible_answers0 = []
    for i in range(N0):
        thinking0, answer0 = await cot_agents0[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents0[i].id}, extract/classify key elements, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_thinkings0.append(thinking0)
        possible_answers0.append(answer0)
    final_decision_agent0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent0([taskInfo] + possible_thinkings0 + possible_answers0, "Sub-task 0.1: Synthesize and choose the most consistent classification of key elements.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent0.id}, classified key elements, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    logs.append({"subtask_id": "subtask_0_1", "instruction": cot_sc_instruction, "agent_collaboration": "SC_CoT", "response": {"thinking": thinking0, "answer": answer0})
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 1.1: Evaluate relevance via Debate
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction1 = "Sub-task 1.1: Evaluate the relevance of each candidate complex (pre-initiation, pre-replication, enhancer, nucleosome histone) to active chromatin in shmoo formation." + debate_instr
    debate_agents1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N1 = self.max_round
    all_thinking1 = [[] for _ in range(N1)]
    all_answer1 = [[] for _ in range(N1)]
    for r in range(N1):
        for agent in debate_agents1:
            if r == 0:
                thinking1, answer1 = await agent([taskInfo, thinking0, answer0], debate_instruction1, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking0, answer0] + all_thinking1[r-1] + all_answer1[r-1]
                thinking1, answer1 = await agent(inputs, debate_instruction1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking1[r].append(thinking1)
            all_answer1[r].append(answer1)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo, thinking0, answer0] + all_thinking1[-1] + all_answer1[-1], "Sub-task 1.1: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent1.id}, relevance evaluation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append({"subtask_id": "subtask_1_1", "instruction": debate_instruction1, "agent_collaboration": "Debate", "response": {"thinking": thinking1, "answer": answer1})
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 1.2: Prioritize complexes via SC-CoT
    cot_sc_instruction2 = "Sub-task 1.2: Prioritize the four complexes by expected enrichment in an active-chromatin ChIP–MS assay during yeast shmoo formation."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    for i in range(N2):
        thinking2, answer2 = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, prioritize complexes, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thinkings2.append(thinking2)
        possible_answers2.append(answer2)
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 1.2: Synthesize and choose the most consistent prioritization of complexes.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent2.id}, prioritization, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append({"subtask_id": "subtask_1_2", "instruction": cot_sc_instruction2, "agent_collaboration": "SC_CoT", "response": {"thinking": thinking2, "answer": answer2})
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 2.1: Determine least observed via CoT
    cot_instruction3 = "Sub-task 2.1: Determine which protein complex will be least observed in the active-chromatin ChIP–MS dataset and justify this selection based on the prioritization."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, determine least observed complex, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append({"subtask_id": "subtask_2_1", "instruction": cot_instruction3, "agent_collaboration": "CoT", "response": {"thinking": thinking3, "answer": answer3})
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
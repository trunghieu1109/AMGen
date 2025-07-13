async def forward_181(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = "Sub-task 1: Identify and summarize the fundamental assumptions and physical conditions under which the Mott-Gurney equation is derived and valid, including device type, carrier type, contact nature, and current components."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking0, answer0 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, summarizing fundamental assumptions, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {
        "thinking": thinking0,
        "answer": answer0
    }
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])

    debate_instr_1 = "Sub-task 1: Analyze the impact of each assumption (trap-free, single-carrier, contact type, diffusion and drift currents) on the applicability of the Mott-Gurney equation, and relate these to the given statements. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1 = self.max_round
    all_thinking1 = [[] for _ in range(N_max_1)]
    all_answer1 = [[] for _ in range(N_max_1)]
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_1,
        "context": ["user query", "thinking of subtask 0", "answer of subtask 0"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking1, answer1 = await agent([taskInfo, thinking0, answer0], debate_instr_1, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo, thinking0, answer0] + all_thinking1[r-1] + all_answer1[r-1]
                thinking1, answer1 = await agent(input_infos_1, debate_instr_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing assumptions, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking1[r].append(thinking1)
            all_answer1[r].append(answer1)
    final_decision_instr_1 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_final, answer1_final = await final_decision_agent_1([taskInfo, thinking0, answer0] + all_thinking1[-1] + all_answer1[-1], "Sub-task 1: Analyze assumptions impact." + final_decision_instr_1, is_sub_task=True)
    agents.append(f"Final Decision agent, analyzing assumptions impact, thinking: {thinking1_final.content}; answer: {answer1_final.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_final.content}; answer - {answer1_final.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1_final,
        "answer": answer1_final
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    debate_instr_2 = "Sub-task 2: Evaluate each of the four given statements against the established validity criteria of the Mott-Gurney equation to determine which statement is true. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking2 = [[] for _ in range(N_max_2)]
    all_answer2 = [[] for _ in range(N_max_2)]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instr_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1_final, answer1_final], debate_instr_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking1_final, answer1_final] + all_thinking2[r-1] + all_answer2[r-1]
                thinking2, answer2 = await agent(input_infos_2, debate_instr_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating statements, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_instr_2 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_final, answer2_final = await final_decision_agent_2([taskInfo, thinking1_final, answer1_final] + all_thinking2[-1] + all_answer2[-1], "Sub-task 2: Evaluate statements." + final_decision_instr_2, is_sub_task=True)
    agents.append(f"Final Decision agent, evaluating statements, thinking: {thinking2_final.content}; answer: {answer2_final.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2_final,
        "answer": answer2_final
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2_final, answer2_final, sub_tasks, agents)
    return final_answer, logs

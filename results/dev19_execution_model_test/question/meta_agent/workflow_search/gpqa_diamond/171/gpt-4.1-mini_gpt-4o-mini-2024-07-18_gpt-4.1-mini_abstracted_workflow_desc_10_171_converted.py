async def forward_171(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Apply the Boltzmann distribution to express the ratio of excited iron atom populations "
        "in star_1 and star_2 in terms of their temperatures and the given energy difference. "
        "Use the given excitation ratio (2) and energy difference (1.38 x 10^-23 J) under LTE assumption."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                               model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, applying Boltzmann distribution, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, combine and manipulate the Boltzmann expressions "
        "to isolate a formula relating ln(2) to T_1 and T_2, incorporating the energy difference and Boltzmann constant. "
        "Consider multiple possible derivations to ensure self-consistency."
    )
    N = self.max_sc
    cot_sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                  model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    possible_thinkings = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_sc_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, manipulating Boltzmann expressions, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2)
        possible_thinkings.append(thinking2)

    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    final_instr_2 = "Given all the above thinking and answers, find the most consistent and correct formula relating ln(2) to T_1 and T_2."
    thinking2, answer2 = await final_decision_agent_2(
        [taskInfo, thinking1, answer1] + possible_thinkings + possible_answers, 
        "Sub-task 2: Synthesize and choose the most consistent formula." + final_instr_2, 
        is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_3 = (
        "Sub-task 3: Evaluate the candidate equations against the derived formula from Sub-task 2 to identify "
        "which equation correctly represents the relationship between T_1 and T_2 given the excitation ratio and energy difference. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                     model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instr_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instr_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instr_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating candidate equations, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)

    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    final_instr_3 = "Given all the above thinking and answers, reason over them carefully and provide a final answer identifying the correct equation."
    thinking3, answer3 = await final_decision_agent_3(
        [taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], 
        "Sub-task 3: Final evaluation and selection of correct equation." + final_instr_3, 
        is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing correct equation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs

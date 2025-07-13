async def forward_197(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Analyze and classify the given chemical data: total cobalt concentration, SCN- concentration, "
        "and stability constants; identify the species involved and clarify the meaning of the stability constants. "
        "Context: The problem involves cobalt(II) thiocyanato complexes with given cumulative stability constants β1=9, β2=40, β3=63, β4=16, total cobalt concentration 10^-2 M, and SCN- concentration 0.1 M."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, analyzing chemical data, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Derive the equilibrium expressions for the concentrations of all cobalt(II) thiocyanato complexes "
        "using the cumulative stability constants and free SCN- concentration. Context: Use the output from stage_0.subtask_1."
    )
    cot_sc_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0, answer_0],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_i, answer_i = await cot_sc_agents_1_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_1[i].id}, deriving equilibrium expressions, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)

    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1_1 = "Given all the above thinking and answers, find the most consistent and correct equilibrium expressions for the cobalt(II) thiocyanato complexes."
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo, thinking_0, answer_0] + possible_thinkings_1_1 + possible_answers_1_1,
        "Sub-task 1: Synthesize and choose the most consistent equilibrium expressions." + final_instr_1_1,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Formulate the mass balance equation for total cobalt and relate it to the sum of all complex species concentrations "
        "to enable solving for free SCN- concentration if needed. Context: Use the output from stage_0.subtask_1."
    )
    cot_sc_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_0, answer_0],
        "agent_collaboration": "Reflexion"
    }
    for i in range(self.max_sc):
        thinking_i, answer_i = await cot_sc_agents_1_2[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_2[i].id}, formulating mass balance, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)

    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1_2 = "Given all the above thinking and answers, find the most consistent and correct mass balance formulation for cobalt."
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2(
        [taskInfo, thinking_0, answer_0] + possible_thinkings_1_2 + possible_answers_1_2,
        "Sub-task 2: Synthesize and choose the most consistent mass balance formulation." + final_instr_1_2,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task stage_1.subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 1: Compute the concentrations of each cobalt(II) thiocyanato complex species, especially the dithiocyanato complex, "
        "by solving the equilibrium and mass balance equations. Context: Use outputs from stage_1.subtask_1 and stage_1.subtask_2."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking_1_1, answer_1_1, thinking_1_2, answer_1_2],
        "agent_collaboration": "SC_CoT"
    }
    thinking_2, answer_2 = await cot_agent_2([taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, computing species concentrations, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_3 = (
        "Sub-task 1: Calculate the percentage of the blue dithiocyanato cobalt(II) complex relative to the total cobalt concentration "
        "and select the closest matching answer choice. Context: Use output from stage_2.subtask_1. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]
    subtask_desc_3 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instr_3,
        "context": ["user query", thinking_2, answer_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking_3, answer_3 = await agent([taskInfo, thinking_2, answer_2], debate_instr_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking_2, answer_2] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking_3, answer_3 = await agent(input_infos_3, debate_instr_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating percentage and selecting answer, thinking: {thinking_3.content}; answer: {answer_3.content}")
            all_thinking_3[r].append(thinking_3)
            all_answer_3[r].append(answer_3)

    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_3 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking_3_final, answer_3_final = await final_decision_agent_3(
        [taskInfo, thinking_2, answer_2] + all_thinking_3[-1] + all_answer_3[-1],
        "Sub-task 1: Calculate percentage and select closest answer choice." + final_instr_3,
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, calculating final percentage and selecting answer, thinking: {thinking_3_final.content}; answer: {answer_3_final.content}")
    sub_tasks.append(f"Sub-task stage_3.subtask_1 output: thinking - {thinking_3_final.content}; answer - {answer_3_final.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3_final, "answer": answer_3_final}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_final, answer_3_final, sub_tasks, agents)
    return final_answer, logs

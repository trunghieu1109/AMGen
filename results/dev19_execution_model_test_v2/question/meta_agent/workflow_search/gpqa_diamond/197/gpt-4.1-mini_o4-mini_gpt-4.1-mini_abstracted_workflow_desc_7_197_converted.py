async def forward_197(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Extract and transform the given chemical data into equilibrium expressions and define the species involved, "
        "including total concentrations and cumulative stability constants, with context from the user query."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, extracting and transforming chemical data, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    debate_instruction_1 = (
        "Sub-task 1: Evaluate the significance of each cobalt(II) thiocyanato complex species based on their stability constants and ligand concentrations "
        "to prioritize which species contribute meaningfully to the equilibrium, given the output from stage_0.subtask_1. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1 = self.max_round
    all_thinking_1 = [[] for _ in range(N_max_1)]
    all_answer_1 = [[] for _ in range(N_max_1)]
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instruction_1,
        "context": ["user query", thinking_0, answer_0],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking_1, answer_1 = await agent([taskInfo, thinking_0, answer_0], debate_instruction_1, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo, thinking_0, answer_0] + all_thinking_1[r-1] + all_answer_1[r-1]
                thinking_1, answer_1 = await agent(input_infos_1, debate_instruction_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating species significance, thinking: {thinking_1.content}; answer: {answer_1.content}")
            all_thinking_1[r].append(thinking_1)
            all_answer_1[r].append(answer_1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1, answer_1 = await final_decision_agent_1([taskInfo, thinking_0, answer_0] + all_thinking_1[-1] + all_answer_1[-1],
                                                      "Sub-task 1: Synthesize and choose the most consistent answer for species significance. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.",
                                                      is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze the equilibrium system by setting up mass balance and equilibrium expressions to calculate the concentration of each cobalt species, "
        "especially the dithiocyanato complex, based on outputs from stage_0.subtask_1 and stage_1.subtask_1."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking_0, answer_0, thinking_1, answer_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2, answer_2 = await cot_agents_2[i]([taskInfo, thinking_0, answer_0, thinking_1, answer_1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing equilibrium system, thinking: {thinking_2.content}; answer: {answer_2.content}")
        possible_answers_2.append(answer_2)
        possible_thinkings_2.append(thinking_2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2, answer_2 = await final_decision_agent_2([taskInfo, thinking_0, answer_0, thinking_1, answer_1] + possible_thinkings_2 + possible_answers_2,
                                                      "Sub-task 2: Synthesize and choose the most consistent and correct solutions for equilibrium concentrations.",
                                                      is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction_3 = (
        "Sub-task 3: Select and calculate the percentage of the dithiocyanato cobalt(II) complex relative to the total cobalt concentration using the concentrations obtained from equilibrium analysis. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]
    subtask_desc_3 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3,
        "context": ["user query", thinking_2, answer_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking_3, answer_3 = await agent([taskInfo, thinking_2, answer_2], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking_2, answer_2] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking_3, answer_3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating percentage of dithiocyanato complex, thinking: {thinking_3.content}; answer: {answer_3.content}")
            all_thinking_3[r].append(thinking_3)
            all_answer_3[r].append(answer_3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3, answer_3 = await final_decision_agent_3([taskInfo, thinking_2, answer_2] + all_thinking_3[-1] + all_answer_3[-1],
                                                      "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
                                                      is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final percentage, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3, answer_3, sub_tasks, agents)
    return final_answer, logs

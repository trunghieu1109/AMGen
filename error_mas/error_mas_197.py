async def forward_197(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Extract and clarify given information and constants
    debate_instr_1 = "Sub-task 1: Extract and summarize all given information from the query, including concentrations, stability constants, and the definition of the target complex. Ensure clarity on the problem context and what is being asked."
    debate_instr_2 = "Sub-task 2: Clarify the nature of the given stability constants β1, β2, β3, and β4 by determining whether they are cumulative or stepwise constants. Derive the correct stepwise stability constants (K1, K2, K3, K4) from the given β values by appropriate division. This subtask explicitly addresses the previous failure of misinterpreting constants and ensures correct equilibrium expressions in later steps."

    debate_agents_stage1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]

    # Subtask 1: Extract and summarize given information
    all_thinking_1 = []
    all_answer_1 = []
    for agent in debate_agents_stage1:
        thinking1, answer1 = await agent([taskInfo], debate_instr_1, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, Subtask 1, thinking: {thinking1.content}; answer: {answer1.content}")
        all_thinking_1.append(thinking1)
        all_answer_1.append(answer1)

    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_final, answer1_final = await final_decision_agent_1([taskInfo] + all_thinking_1 + all_answer_1, "Sub-task 1: Synthesize and finalize extraction and summary of given information.", is_sub_task=True)
    agents.append(f"Final Decision agent Subtask 1, thinking: {thinking1_final.content}; answer: {answer1_final.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_final.content}; answer - {answer1_final.content}")
    logs.append({"subtask_id": "stage_1.subtask_1", "instruction": debate_instr_1, "response": {"thinking": thinking1_final, "answer": answer1_final})
    print("Step 1: ", sub_tasks[-1])

    # Subtask 2: Clarify nature of stability constants and derive stepwise constants
    all_thinking_2 = []
    all_answer_2 = []
    for agent in debate_agents_stage1:
        thinking2, answer2 = await agent([taskInfo, thinking1_final, answer1_final], debate_instr_2, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, Subtask 2, thinking: {thinking2.content}; answer: {answer2.content}")
        all_thinking_2.append(thinking2)
        all_answer_2.append(answer2)

    thinking2_final, answer2_final = await final_decision_agent_1([taskInfo, thinking1_final, answer1_final] + all_thinking_2 + all_answer_2, "Sub-task 2: Synthesize and finalize derivation of stepwise stability constants.", is_sub_task=True)
    agents.append(f"Final Decision agent Subtask 2, thinking: {thinking2_final.content}; answer: {answer2_final.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final.content}")
    logs.append({"subtask_id": "stage_1.subtask_2", "instruction": debate_instr_2, "response": {"thinking": thinking2_final, "answer": answer2_final})
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Write equilibrium expressions and formulate mass balances using SC_CoT
    cot_sc_instruction_1 = "Sub-task 1: Write the equilibrium expressions for all cobalt(II) thiocyanato complexes using the correctly derived stepwise stability constants and initial concentrations. Explicitly formulate the expressions for the concentrations of free Co(II), mono-, di-, tri-, and tetrathiocyanato complexes in terms of free SCN- concentration and total Co concentration. Avoid previous errors of using cumulative constants directly."
    cot_sc_instruction_2 = "Sub-task 2: Formulate the mass balance equations for cobalt and thiocyanate ions, incorporating the equilibrium expressions from subtask_2.1. Include the cobalt mass balance (sum of all Co species equals total Co) and the thiocyanate mass balance (free SCN- plus SCN- bound in complexes equals initial SCN- concentration). This addresses the previous neglect of SCN- mass balance and ensures accurate free ligand concentration estimation."

    N = self.max_sc
    cot_agents_stage2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]

    possible_answers_21 = []
    possible_thinkings_21 = []
    for i in range(N):
        thinking21, answer21 = await cot_agents_stage2[i]([taskInfo, thinking2_final, answer2_final], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage2[i].id}, Subtask 2.1, thinking: {thinking21.content}; answer: {answer21.content}")
        possible_thinkings_21.append(thinking21)
        possible_answers_21.append(answer21)

    final_decision_agent_21 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking21_final, answer21_final = await final_decision_agent_21([taskInfo, thinking2_final, answer2_final] + possible_thinkings_21 + possible_answers_21, "Sub-task 2.1: Synthesize and finalize equilibrium expressions.", is_sub_task=True)
    agents.append(f"Final Decision agent Subtask 2.1, thinking: {thinking21_final.content}; answer: {answer21_final.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking21_final.content}; answer - {answer21_final.content}")
    logs.append({"subtask_id": "stage_2.subtask_1", "instruction": cot_sc_instruction_1, "response": {"thinking": thinking21_final, "answer": answer21_final})
    print("Step 3: ", sub_tasks[-1])

    possible_answers_22 = []
    possible_thinkings_22 = []
    for i in range(N):
        thinking22, answer22 = await cot_agents_stage2[i]([taskInfo, thinking2_final, answer2_final, thinking21_final, answer21_final], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage2[i].id}, Subtask 2.2, thinking: {thinking22.content}; answer: {answer22.content}")
        possible_thinkings_22.append(thinking22)
        possible_answers_22.append(answer22)

    thinking22_final, answer22_final = await final_decision_agent_21([taskInfo, thinking2_final, answer2_final, thinking21_final, answer21_final] + possible_thinkings_22 + possible_answers_22, "Sub-task 2.2: Synthesize and finalize mass balance equations.", is_sub_task=True)
    agents.append(f"Final Decision agent Subtask 2.2, thinking: {thinking22_final.content}; answer: {answer22_final.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking22_final.content}; answer - {answer22_final.content}")
    logs.append({"subtask_id": "stage_2.subtask_2", "instruction": cot_sc_instruction_2, "response": {"thinking": thinking22_final, "answer": answer22_final})
    print("Step 4: ", sub_tasks[-1])

    # Stage 3: Numerically solve system and compute fraction of dithiocyanato complex using Debate
    debate_instr_3 = "Sub-task 1: Numerically solve the system of equilibrium and mass balance equations to determine the free SCN- concentration and the concentrations of all cobalt(II) species, especially the blue dithiocyanato complex. Explicitly compute the fraction of the dithiocyanato complex using the formula: fraction = β2 * [SCN-]^2 / (1 + β1 * [SCN-] + β2 * [SCN-]^2 + β3 * [SCN-]^3 + β4 * [SCN-]^4), where β values are cumulative constants. This subtask corrects the previous failure of not performing explicit numeric speciation and avoids assumptions. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."

    debate_agents_stage3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round

    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]

    subtask_desc_3 = {"subtask_id": "stage_3.subtask_1", "instruction": debate_instr_3, "context": ["user query", thinking22_final.content, answer22_final.content], "agent_collaboration": "Debate"}

    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_stage3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking22_final, answer22_final], debate_instr_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking22_final, answer22_final] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instr_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_3[r].append(thinking3)
            all_answer_3[r].append(answer3)

    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_final, answer3_final = await final_decision_agent_3([taskInfo, thinking22_final, answer22_final] + all_thinking_3[-1] + all_answer_3[-1], "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent Subtask 3, thinking: {thinking3_final.content}; answer: {answer3_final.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final.content}")
    logs.append(subtask_desc_3)
    print("Step 5: ", sub_tasks[-1])

    # Stage 4: Calculate percentage and select closest choice using Debate
    debate_instr_4 = "Sub-task 1: Calculate the percentage of the blue dithiocyanato cobalt(II) complex relative to the total cobalt concentration based on the numeric results from stage_3. Compare the calculated percentage with the given choices and select the closest matching option. This final step ensures the answer is grounded in correct calculations rather than assumptions. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."

    debate_agents_stage4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round

    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]

    subtask_desc_4 = {"subtask_id": "stage_4.subtask_1", "instruction": debate_instr_4, "context": ["user query", thinking3_final.content, answer3_final.content], "agent_collaboration": "Debate"}

    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_stage4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3_final, answer3_final], debate_instr_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3_final, answer3_final] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instr_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking_4[r].append(thinking4)
            all_answer_4[r].append(answer4)

    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4_final, answer4_final = await final_decision_agent_4([taskInfo, thinking3_final, answer3_final] + all_thinking_4[-1] + all_answer_4[-1], "Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent Subtask 4, thinking: {thinking4_final.content}; answer: {answer4_final.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4_final.content}; answer - {answer4_final.content}")
    logs.append(subtask_desc_4)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4_final, answer4_final, sub_tasks, agents)
    return final_answer, logs

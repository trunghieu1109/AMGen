async def forward_8(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Formalize game elements using Chain-of-Thought and Self-Consistency
    cot_instruction_0 = (
        "Sub-task 1: Formally represent the game elements: define the state space as the number of tokens n with 1 ≤ n ≤ 2024. "
        "Specify the allowed moves (removing 1 or 4 tokens) and turn order (Alice first). Define the winning condition clearly (player who removes the last token wins). "
        "Establish the concepts of winning and losing positions from the perspective of the current player. Clarify that Bob can guarantee a win if the initial position is losing for Alice. "
        "Explicitly state assumptions: perfect play, no chance, moves allowed only if enough tokens remain. Avoid any assumptions beyond standard impartial combinatorial game theory rules.")
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, formalizing game rules, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Stage 0 Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)

    # Stage 1 Sub-task 1: Enumerate winning/losing positions for n=0..15 using CoT and Self-Consistency
    cot_sc_instruction_1 = (
        "Sub-task 1: Enumerate and tabulate the winning and losing positions for all n from 0 up to 15. "
        "Use the formal definitions from Stage 0. For each position n, determine if it is winning or losing for the current player using the standard approach: "
        "a position is losing if all moves lead to winning positions; winning if there exists at least one move to a losing position. "
        "Explicitly produce a table or list of results. This enumeration will serve as the empirical foundation for pattern derivation. Avoid skipping this step or making assumptions without verification.")
    N_sc = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1, answer_1 = await cot_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, enumerating positions n=0..15, thinking: {thinking_1.content}; answer: {answer_1.content}")
        possible_answers_1.append(answer_1)
        possible_thinkings_1.append(thinking_1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1, answer_1 = await final_decision_agent_1([taskInfo] + possible_answers_1 + possible_thinkings_1, 
                                                      "Sub-task 1: Synthesize and choose the most consistent enumeration table for n=0..15.", 
                                                      is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 1 Sub-task 2: Derive and rigorously verify losing position pattern using Reflexion
    reflect_inst_2 = (
        "Sub-task 2: Derive a candidate closed-form or modular arithmetic pattern describing all losing positions for the first player (Alice) based on the enumeration table from subtask_1. "
        "Rigorously check the pattern against every enumerated position to ensure no losing positions are missed (e.g., verify if positions with n mod 5 = 2 are losing or not). "
        "Provide explicit reasoning or proof linking the pattern to the game rules and enumeration results. Avoid partial or ambiguous characterizations. "
        "Clearly state that losing positions for Alice correspond exactly to positions where Bob, moving second, can guarantee a win. Exclude n=0 from the count as it is not positive.")
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2 = self.max_round
    cot_inputs_2 = [taskInfo, thinking_1, answer_1]
    subtask_desc_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": reflect_inst_2,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2, answer_2 = await cot_agent_2(cot_inputs_2, reflect_inst_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, deriving and verifying pattern, thinking: {thinking_2.content}; answer: {answer_2.content}")
    for i in range(N_max_2):
        feedback_2, correct_2 = await critic_agent_2([taskInfo, thinking_2, answer_2], 
                                                   "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", 
                                                   i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, feedback: {feedback_2.content}; correct: {correct_2.content}")
        if correct_2.content == "True":
            break
        cot_inputs_2.extend([thinking_2, answer_2, feedback_2])
        thinking_2, answer_2 = await cot_agent_2(cot_inputs_2, reflect_inst_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining pattern, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 1 Sub-task 3: Conflict resolution and debate among agents to reconcile contradictory patterns
    debate_instr_3 = (
        "Sub-task 3: Conduct a dedicated conflict resolution and debate session among agents to reconcile any contradictory patterns or conclusions about losing positions (e.g., n mod 5 = 0 only vs. n mod 5 = 0 or 2). "
        "Use explicit counterexamples from the enumeration table to argue for or against competing patterns. Reach consensus on the correct pattern and document this agreement clearly. "
        "This step ensures that downstream subtasks have a verified and uncontested foundation.")
    debate_instruction_3 = "Sub-task 3: Your problem is to reconcile conflicting losing position patterns." + " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]
    subtask_desc_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", thinking_1.content, answer_1.content, thinking_2.content, answer_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking_3, answer_3 = await agent([taskInfo, thinking_1, answer_1, thinking_2, answer_2], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking_1, answer_1, thinking_2, answer_2] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking_3, answer_3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_3.content}; answer: {answer_3.content}")
            all_thinking_3[r].append(thinking_3)
            all_answer_3[r].append(answer_3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3, answer_3 = await final_decision_agent_3([taskInfo] + all_thinking_3[-1] + all_answer_3[-1], 
                                                      "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.", 
                                                      is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2 Sub-task 1: Count number of positive integers n ≤ 2024 where Bob can guarantee a win using verified pattern
    cot_instruction_4 = (
        "Sub-task 1: Using the verified pattern for losing positions from stage_1.subtask_3, compute the total count of positive integers n ≤ 2024 for which the initial position is losing for Alice. "
        "This count represents the number of n for which Bob has a guaranteed winning strategy. Perform the counting carefully using modular arithmetic or direct formula, ensuring no off-by-one or boundary errors. Avoid brute force enumeration for all n unless justified by computational simplicity.")
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_4 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking_3.content, answer_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_4, answer_4 = await cot_agent_4([taskInfo, thinking_3, answer_3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, counting losing positions up to 2024, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 2 Sub-task 2: Verify final count by cross-checking with sample values and pattern
    cot_instruction_5 = (
        "Sub-task 2: Verify the final count by cross-checking with sample values from the enumeration table and the derived pattern. "
        "Confirm that the result aligns with the theoretical analysis and no edge cases (such as n=0 or boundary values near 2024) are missed. "
        "Provide the final answer explicitly and summarize the reasoning behind it, including the equivalence between losing positions for Alice and Bob's guaranteed wins. "
        "Include a brief explanation of why the pattern holds for all n ≤ 2024.")
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_5 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking_4.content, answer_4.content, thinking_1.content, answer_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_5, answer_5 = await cot_agent_5([taskInfo, thinking_4, answer_4, thinking_1, answer_1], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, verifying final count, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Stage 2 Sub-task 2 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_5, answer_5, sub_tasks, agents)
    return final_answer, logs
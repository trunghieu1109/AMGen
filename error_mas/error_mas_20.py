import math
from collections import Counter

async def forward_20(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = (
        "Sub-task 1: Define the digit constraints for two-digit numbers in base b. "
        "Explicitly state the ranges for digits x and y, ensuring the leading digit x satisfies 1 ≤ x ≤ b-1 and the second digit y satisfies 0 ≤ y ≤ b-1. "
        "Emphasize that n = x*b + y represents a two-digit number in base b with no leading zero."
    )
    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking1, answer1 = await cot_sc_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, defining digit constraints, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings_1, "Sub-task 1: Synthesize and finalize digit constraints definition.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Formulate the mathematical condition for a number n = x*b + y to be b-eautiful: "
        "that is, the sum of digits x + y equals the integer square root of n, and n is a perfect square. "
        "Explicitly state that √n must be an integer and equal to x + y."
    )
    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_sc_agents_2[i]([taskInfo, thinking1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, formulating b-eautiful condition, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_thinkings_2, "Sub-task 2: Synthesize and finalize b-eautiful condition.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 3: Clarify and confirm all assumptions and constraints: n is a positive integer, digits satisfy 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1, "
        "and √n = x + y is an integer. Reiterate that n must have exactly two digits in base b and that the problem requires counting such n for varying b."
    )
    cot_sc_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking1.content, thinking2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking3, answer3 = await cot_sc_agents_3[i]([taskInfo, thinking1, thinking2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3[i].id}, confirming assumptions, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + possible_thinkings_3, "Sub-task 3: Synthesize and finalize assumptions and constraints.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: For a fixed base b, enumerate all possible digit pairs (x,y) with x in [1, b-1] and y in [0, b-1]. "
        "Generate the list of candidate two-digit numbers n = x*b + y to be tested for b-eautifulness."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, enumerating digit pairs for fixed base, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = (
        "Sub-task 5: For each digit pair (x,y) enumerated in subtask_4, compute n = x*b + y and check if n is a perfect square. "
        "If so, verify whether x + y equals √n. Collect all digit pairs (x,y) and corresponding n that satisfy these conditions."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, checking perfect squares and digit sums, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6 = (
        "Sub-task 6: Count the total number of b-eautiful integers for the given base b by aggregating all valid (x,y) pairs found in subtask_5. "
        "Output this count explicitly as a numeric value."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", thinking5.content],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, counting b-eautiful integers, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    debate_instruction_7 = (
        "Sub-task 7: Implement a systematic enumeration over bases b starting from 2 up to 30. "
        "For each base b, perform subtasks 4 to 6 to compute the exact count of b-eautiful integers. "
        "Store and output the counts for all tested bases as explicit numeric data. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": debate_instruction_7,
        "context": ["user query", thinking6.content],
        "agent_collaboration": "Debate"
    }

    def enumerate_b_eautiful_counts(b):
        count = 0
        for x in range(1, b):
            for y in range(0, b):
                n = x * b + y
                s = x + y
                if s * s == n:
                    count += 1
        return count

    enumeration_results = {}
    for b in range(2, 31):
        enumeration_results[b] = enumerate_b_eautiful_counts(b)

    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking6] + all_thinking7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, enumerating bases and counts, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)

    final_decision_instruction_7 = (
        "Sub-task 7: Given all the above thinking and answers, reason over them carefully and provide a final enumeration result. "
        "Output the dictionary of base to count of b-eautiful integers explicitly."
    )
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1], final_decision_instruction_7, is_sub_task=True)
    agents.append(f"Final Decision agent, enumerating bases and counts, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7, "enumeration_results": enumeration_results}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    debate_instruction_8 = (
        "Sub-task 8: Analyze the enumeration results from subtask_7 to identify the smallest base b ≥ 2 for which the count of b-eautiful integers exceeds ten. "
        "This decision must be based solely on the verified numeric counts, avoiding heuristic or speculative reasoning. "
        "Output this minimal base b as the final answer. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_8 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_8 = self.max_round
    all_thinking8 = [[] for _ in range(N_max_8)]
    all_answer8 = [[] for _ in range(N_max_8)]
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": debate_instruction_8,
        "context": ["user query", thinking7.content, str(enumeration_results)],
        "agent_collaboration": "Debate"
    }

    minimal_base = None
    for b in range(2, 31):
        if enumeration_results[b] > 10:
            minimal_base = b
            break

    for r in range(N_max_8):
        for i, agent in enumerate(debate_agents_8):
            if r == 0:
                thinking8, answer8 = await agent([taskInfo, thinking7, str(enumeration_results), f"Minimal base with count > 10: {minimal_base}"], debate_instruction_8, r, is_sub_task=True)
            else:
                input_infos_8 = [taskInfo, thinking7, str(enumeration_results), f"Minimal base with count > 10: {minimal_base}"] + all_thinking8[r-1]
                thinking8, answer8 = await agent(input_infos_8, debate_instruction_8, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, deciding minimal base, thinking: {thinking8.content}; answer: {answer8.content}")
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)

    final_decision_instruction_8 = (
        "Sub-task 8: Given all the above thinking and answers, reason over them carefully and provide the minimal base b ≥ 2 for which the count of b-eautiful integers exceeds ten. "
        "Output this minimal base b as the final answer."
    )
    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent_8([taskInfo] + all_thinking8[-1], final_decision_instruction_8, is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing minimal base, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8, "minimal_base": minimal_base}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    reflect_inst_9 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_9 = (
        "Sub-task 9: Perform a verification and reflexion phase to confirm the accuracy of the counts obtained in subtask_7. "
        "Cross-check the enumeration logic, validate the perfect square and digit sum conditions, and ensure no valid b-eautiful integers are missed or double-counted. "
        "Confirm the correctness of the minimal base b identified in subtask_8 before finalizing the answer. "
        + reflect_inst_9
    )
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_9 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_9 = self.max_round
    cot_inputs_9 = [taskInfo, thinking7, thinking8]
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_reflect_instruction_9,
        "context": ["user query", thinking7.content, thinking8.content],
        "agent_collaboration": "Reflexion"
    }
    thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_9.id}, verifying enumeration and minimal base, thinking: {thinking9.content}; answer: {answer9.content}")
    for i in range(N_max_9):
        feedback9, correct9 = await critic_agent_9([taskInfo, thinking9], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_9.id}, providing feedback, thinking: {feedback9.content}; answer: {correct9.content}")
        if correct9.content.strip() == "True":
            break
        cot_inputs_9.extend([thinking9, feedback9])
        thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_9.id}, refining verification, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs

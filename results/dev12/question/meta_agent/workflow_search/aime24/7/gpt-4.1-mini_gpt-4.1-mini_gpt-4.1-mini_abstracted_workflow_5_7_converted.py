async def forward_7(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    cot_sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]

    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]

    # Stage 1: Transform logarithmic equations into exponential form

    cot_instruction_1 = (
        "Sub-task 1: Transform the first logarithmic equation log_x(y^x) = 10 into its equivalent exponential form. "
        "Clearly express the resulting equation and emphasize the domain constraints x > 1, y > 1. Avoid any assumptions or simplifications at this stage."
    )
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, transforming first logarithmic equation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Transform the second logarithmic equation log_y(x^{4y}) = 10 into its equivalent exponential form. "
        "Clearly express the resulting equation and emphasize the domain constraints x > 1, y > 1. Avoid any assumptions or simplifications at this stage."
    )
    subtask_desc2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, transforming second logarithmic equation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 3: Summarize the two exponential equations obtained from the previous subtasks. "
        "Restate the problem in terms of these equations, explicitly noting the domain constraints and the transcendental nature of the system. "
        "Avoid attempting to solve or simplify the system here."
    )
    subtask_desc3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers = []
    possible_thinkings = []
    for i in range(self.max_sc):
        thinking3, answer3 = await cot_sc_agents[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, summarizing exponential equations, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers.append(answer3)
        possible_thinkings.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + possible_answers + possible_thinkings, "Sub-task 3: Synthesize and choose the most consistent summary of exponential equations.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Apply natural logarithms and derive key relation xy=25

    cot_instruction_4 = (
        "Sub-task 1: Apply natural logarithms to both exponential equations to convert them into equations involving ln(x) and ln(y). "
        "Clearly write down these transformed equations, ensuring all steps are explicit and justified. Avoid premature numeric substitution or assumptions."
    )
    subtask_desc4 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, applying natural logarithms, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = (
        "Sub-task 2: From the logarithmic forms, algebraically manipulate the system to isolate one variable in terms of the other or to derive a direct relation between x and y. "
        "Explicitly derive the key relation xy = 25 or an equivalent symbolic expression, providing rigorous justification for each step. Avoid unjustified simplifications or assumptions."
    )
    subtask_desc5 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, deriving key relation xy=25, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_sc_instruction_6 = (
        "Sub-task 3: Using the relation derived in the previous subtask, substitute back into one of the logarithmic equations to solve for x or y explicitly or numerically. "
        "Ensure that the transcendental nature of the system is properly handled, and avoid relying solely on numeric root searches without symbolic context."
    )
    subtask_desc6 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", thinking5.content, answer5.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers6 = []
    possible_thinkings6 = []
    for i in range(self.max_sc):
        thinking6, answer6 = await cot_sc_agents[i]([taskInfo, thinking5, answer5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, solving for x or y numerically/symbolically, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers6.append(answer6)
        possible_thinkings6.append(thinking6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + possible_answers6 + possible_thinkings6, "Sub-task 6: Synthesize and choose the most consistent solution for x or y.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_7 = (
        "Sub-task 4: Compute the product xy using the values or expressions obtained. "
        "Clearly show the calculation steps and ensure consistency with the derived relations and domain constraints x > 1, y > 1."
    )
    subtask_desc7 = {
        "subtask_id": "stage_2.subtask_4",
        "instruction": cot_instruction_7,
        "context": ["user query", thinking6.content, answer6.content],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing product xy, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    debate_instruction_8 = (
        "Sub-task 5: Verify the candidate solution(s) by substituting the values of x and y back into the original logarithmic equations. "
        "Check that both equations hold true within acceptable tolerance and that the domain constraints are satisfied. "
        "Discuss the uniqueness or multiplicity of solutions and confirm the correctness of numeric approximations if any. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    subtask_desc8 = {
        "subtask_id": "stage_2.subtask_5",
        "instruction": debate_instruction_8,
        "context": ["user query", thinking7.content, answer7.content],
        "agent_collaboration": "Debate"
    }
    N_max_8 = self.max_round
    all_thinking8 = [[] for _ in range(N_max_8)]
    all_answer8 = [[] for _ in range(N_max_8)]
    for r in range(N_max_8):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking8, answer8 = await agent([taskInfo, thinking7, answer7], debate_instruction_8, r, is_sub_task=True)
            else:
                input_infos_8 = [taskInfo, thinking7, answer7] + all_thinking8[r-1] + all_answer8[r-1]
                thinking8, answer8 = await agent(input_infos_8, debate_instruction_8, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying candidate solutions, thinking: {thinking8.content}; answer: {answer8.content}")
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)
    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent_8([taskInfo] + all_thinking8[-1] + all_answer8[-1], "Sub-task 8: Given all the above thinking and answers, reason over them carefully and provide a final verified answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 6: Reflect on the entire solution process, summarizing the key findings and confirming the final value of xy. " + reflect_inst
    subtask_desc9 = {
        "subtask_id": "stage_2.subtask_6",
        "instruction": cot_reflect_instruction,
        "context": ["user query", thinking8.content, answer8.content],
        "agent_collaboration": "Reflexion"
    }
    cot_reflect_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs = [taskInfo, thinking8, answer8]
    thinking9, answer9 = await cot_reflect_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_reflect_agent.id}, reflecting on solution, thinking: {thinking9.content}; answer: {answer9.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking9, answer9], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking9, answer9, feedback])
        thinking9, answer9 = await cot_reflect_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_reflect_agent.id}, refining solution, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs

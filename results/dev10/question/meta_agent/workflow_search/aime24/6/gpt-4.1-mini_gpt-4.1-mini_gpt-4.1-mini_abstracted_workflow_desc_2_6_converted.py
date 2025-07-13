async def forward_6(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Formally define the problem variables and constraints without any assumptions on edge ordering or equality. "
        "Specifically: (a) Define variables x, y, z > 0 representing the edges of the rectangular box; "
        "(b) express the surface area constraint as 2(xy + yz + zx) = 54; "
        "(c) express the volume constraint as xyz = 23; "
        "(d) define the squared space diagonal d^2 = x^2 + y^2 + z^2; "
        "(e) relate the minimal enclosing sphere radius squared r^2 = d^2 / 4. "
        "Emphasize the need to keep all variables positive real numbers and avoid any assumptions about symmetry or ordering at this stage. "
        "Provide a structured summary of constraints and the objective function for optimization."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, formal problem definition, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Formulate the constrained optimization problem to maximize d^2 = x^2 + y^2 + z^2 subject to the constraints "
        "2(xy + yz + zx) = 54 and xyz = 23 with x, y, z > 0. Construct the Lagrangian function incorporating two multipliers. "
        "Derive the full system of equations by setting partial derivatives of the Lagrangian with respect to x, y, and z to zero, alongside the original constraints. "
        "Present the system explicitly without simplifying assumptions. "
        "Find all real positive solutions (x, y, z) to this system, including asymmetric ones. "
        "Provide numeric approximations of each candidate solution with at least 6 decimal places. "
        "Structure outputs as a list of candidates with their corresponding (x, y, z) values. "
        "Do not assume symmetry unless rigorously justified."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1.content, answer1.content], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, solve Lagrange system, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent and complete set of candidate solutions for the Lagrange multiplier system, including numeric approximations and no premature symmetry assumptions.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 3: For each candidate solution from Sub-task 2, perform detailed verification: "
        "(a) numerically verify that the surface area and volume constraints hold within a tolerance of 10^-6; "
        "(b) compute the squared diagonal d^2 = x^2 + y^2 + z^2; "
        "(c) check second-order conditions or perform perturbation tests to confirm whether the candidate corresponds to a local maximum of d^2; "
        "(d) discard solutions that violate constraints or do not correspond to maxima; "
        "(e) tabulate verified candidates with their d^2 values and structured numeric data."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2.content, answer2.content], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, verify candidates, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + possible_answers_3 + possible_thinkings_3, "Sub-task 3: Synthesize verified candidates that satisfy constraints and correspond to local maxima, with numeric data.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Identify the candidate solution with the maximum squared diagonal d^2 among the verified candidates from Sub-task 3. "
        "Compute the minimal enclosing sphere radius squared r^2 = d^2 / 4. "
        "Express r^2 as a reduced fraction p/q, where p and q are positive integers with gcd(p, q) = 1. "
        "Provide the final answer p + q. "
        "Include a detailed justification that the chosen candidate yields the global maximum based on previous verification steps."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3.content, answer3.content], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, select max candidate and compute final answer, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instr_5 = (
        "Sub-task 5: Given solutions to the problem from other agents, consider their opinions as additional advice. "
        "Please think carefully and provide an updated answer. "
        "Critically verify all results from Sub-tasks 2 to 4 via a multi-pass process: "
        "(a) confirm that all candidate solutions satisfy the first-order conditions and constraints; "
        "(b) verify maximality rigorously using second-order conditions or perturbation arguments; "
        "(c) challenge any assumption of symmetry or uniqueness by searching for alternative candidates; "
        "(d) confirm correctness of the reduced fraction form and final sum p + q; "
        "(e) provide a reasoned acceptance or rejection of the final answer, including identification of any unresolved doubts or assumptions. "
        "This subtask acts as a rigorous quality control and debate step before finalizing the solution."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instr_5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4.content, answer4.content], debate_instr_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4.content, answer4.content] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instr_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    reflect_inst_6 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_6 = "Sub-task 6: Critically verify all results from subtasks 3 to 5, challenge assumptions, and confirm correctness." + reflect_inst_6
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking3.content, answer3.content, thinking4.content, answer4.content, thinking5.content, answer5.content]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", thinking3.content, answer3.content, thinking4.content, answer4.content, thinking5.content, answer5.content],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, initial verification, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6.content, answer6.content], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6.content, answer6.content, feedback.content])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refinement round {i+1}, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    reflect_inst_7 = (
        "Sub-task 7: Reflect on the entire solution process and results: "
        "(a) explicitly analyze and question all assumptions made during the solving process, especially regarding symmetry and uniqueness of solutions; "
        "(b) suggest alternative solving approaches or checks that could be performed to confirm global optimality; "
        "(c) discuss implications if any assumptions were incorrect or if alternative solutions exist; "
        "(d) provide final confirmation of the answer or recommend further investigation steps if necessary. "
        "This step ensures robustness and transparency of the final solution."
    )
    cot_reflect_instruction_7 = "Sub-task 7: Reflection and robustness check." + reflect_inst_7
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", thinking6.content, answer6.content, thinking1.content, answer1.content, thinking2.content, answer2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6.content, answer6.content, thinking1.content, answer1.content, thinking2.content, answer2.content], cot_reflect_instruction_7, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, final reflection, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs

async def forward_8(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    cot_sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]

    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)

    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]

    # Stage 0

    # Subtask 1: Formally define the game setup (CoT | SC_CoT)
    cot_instruction_0_1 = "Sub-task 1: Formally define the game setup: two players alternate removing either 1 or 4 tokens from a stack of n tokens, with Alice moving first, and the player removing the last token wins. Clearly state the domain of n as positive integers up to 2024."
    thinking_0_1, answer_0_1 = await cot_agent([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, defining game setup, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")

    cot_sc_instruction_0_1 = "Sub-task 1 (SC): Based on the initial definition, consider multiple perspectives to ensure completeness and clarity of the game setup."
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    for i in range(self.max_sc):
        thinking_sc, answer_sc = await cot_sc_agents[i]([taskInfo, thinking_0_1], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, refining game setup, thinking: {thinking_sc.content}; answer: {answer_sc.content}")
        possible_answers_0_1.append(answer_sc)
        possible_thinkings_0_1.append(thinking_sc)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1_final, answer_0_1_final = await final_decision_agent_0_1([taskInfo] + possible_thinkings_0_1, "Sub-task 1: Synthesize and finalize the game setup definition.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1_final.content}; answer - {answer_0_1_final.content}")
    logs.append({"subtask_id": "stage_0.subtask_1", "instruction": cot_instruction_0_1, "response": {"thinking": thinking_0_1_final, "answer": answer_0_1_final}, "agent_collaboration": "CoT | SC_CoT"})
    print("Step 0.1: ", sub_tasks[-1])

    # Subtask 2: Define winning and losing positions (depends on subtask 1) (CoT | SC_CoT)
    cot_instruction_0_2 = "Sub-task 2: Identify and define the concepts of winning and losing positions in the context of this game, emphasizing that a position is winning if the current player can force a win and losing otherwise."
    thinking_0_2, answer_0_2 = await cot_agent([taskInfo, thinking_0_1_final], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, defining winning/losing positions, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")

    cot_sc_instruction_0_2 = "Sub-task 2 (SC): Consider multiple explanations and examples to clarify winning and losing positions."
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    for i in range(self.max_sc):
        thinking_sc, answer_sc = await cot_sc_agents[i]([taskInfo, thinking_0_1_final, thinking_0_2], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, refining winning/losing definitions, thinking: {thinking_sc.content}; answer: {answer_sc.content}")
        possible_answers_0_2.append(answer_sc)
        possible_thinkings_0_2.append(thinking_sc)
    thinking_0_2_final, answer_0_2_final = await final_decision_agent_0_1([taskInfo] + possible_thinkings_0_2, "Sub-task 2: Synthesize and finalize winning/losing position definitions.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2_final.content}; answer - {answer_0_2_final.content}")
    logs.append({"subtask_id": "stage_0.subtask_2", "instruction": cot_instruction_0_2, "response": {"thinking": thinking_0_2_final, "answer": answer_0_2_final}, "agent_collaboration": "CoT | SC_CoT"})
    print("Step 0.2: ", sub_tasks[-1])

    # Subtask 3: Explain relationship between initial position and players' chances (depends on subtask 2) (CoT | SC_CoT)
    cot_instruction_0_3 = "Sub-task 3: Explain the relationship between the initial position and the players' chances: since Alice moves first, Bob can guarantee a win if and only if the initial position is losing for Alice."
    thinking_0_3, answer_0_3 = await cot_agent([taskInfo, thinking_0_2_final], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, explaining initial position relation, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")

    cot_sc_instruction_0_3 = "Sub-task 3 (SC): Consider multiple ways to explain the initial position's impact on winning chances."
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    for i in range(self.max_sc):
        thinking_sc, answer_sc = await cot_sc_agents[i]([taskInfo, thinking_0_2_final, thinking_0_3], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, refining explanation of initial position, thinking: {thinking_sc.content}; answer: {answer_sc.content}")
        possible_answers_0_3.append(answer_sc)
        possible_thinkings_0_3.append(thinking_sc)
    thinking_0_3_final, answer_0_3_final = await final_decision_agent_0_1([taskInfo] + possible_thinkings_0_3, "Sub-task 3: Synthesize and finalize explanation of initial position.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3_final.content}; answer - {answer_0_3_final.content}")
    logs.append({"subtask_id": "stage_0.subtask_3", "instruction": cot_instruction_0_3, "response": {"thinking": thinking_0_3_final, "answer": answer_0_3_final}, "agent_collaboration": "CoT | SC_CoT"})
    print("Step 0.3: ", sub_tasks[-1])

    # Stage 1

    # Subtask 1: Formulate recurrence for winning/losing positions (depends on stage_0.subtask_2) (CoT | SC_CoT | Reflexion)
    cot_instruction_1_1 = "Sub-task 1: Formulate the recurrence or functional relationship that determines whether a position n is winning or losing based on the positions reachable by removing 1 or 4 tokens."
    thinking_1_1, answer_1_1 = await cot_agent([taskInfo, thinking_0_2_final], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, formulating recurrence, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")

    cot_sc_instruction_1_1 = "Sub-task 1 (SC): Consider multiple formulations and verify correctness of the recurrence relation."
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    for i in range(self.max_sc):
        thinking_sc, answer_sc = await cot_sc_agents[i]([taskInfo, thinking_0_2_final, thinking_1_1], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, refining recurrence, thinking: {thinking_sc.content}; answer: {answer_sc.content}")
        possible_answers_1_1.append(answer_sc)
        possible_thinkings_1_1.append(thinking_sc)

    thinking_1_1_final, answer_1_1_final = await final_decision_agent_0_1([taskInfo] + possible_thinkings_1_1, "Sub-task 1: Synthesize and finalize recurrence relation.", is_sub_task=True)

    reflect_inst_1_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_1 = "Sub-task 1 (Reflexion): Your problem is to formulate the recurrence for winning/losing positions." + reflect_inst_1_1

    thinking_reflect, answer_reflect = await cot_agent([taskInfo] + possible_thinkings_1_1 + [thinking_1_1_final], cot_reflect_instruction_1_1, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, refining recurrence, thinking: {thinking_reflect.content}; answer: {answer_reflect.content}")

    critic_inst_1_1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking_reflect], critic_inst_1_1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        thinking_reflect, answer_reflect = await cot_agent([taskInfo, thinking_reflect, feedback], cot_reflect_instruction_1_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining recurrence, thinking: {thinking_reflect.content}; answer: {answer_reflect.content}")

    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_reflect.content}; answer - {answer_reflect.content}")
    logs.append({"subtask_id": "stage_1.subtask_1", "instruction": cot_instruction_1_1, "response": {"thinking": thinking_reflect, "answer": answer_reflect}, "agent_collaboration": "CoT | SC_CoT | Reflexion"})
    print("Step 1.1: ", sub_tasks[-1])

    # Subtask 2: Compute and tabulate winning/losing status for n=1 to 10 (depends on stage_1.subtask_1) (CoT | SC_CoT | Reflexion)
    cot_instruction_1_2 = "Sub-task 2: Compute and tabulate the winning/losing status for the first several values of n (1 to 10) using the recurrence to identify any emerging patterns or periodicity."
    thinking_1_2, answer_1_2 = await cot_agent([taskInfo, thinking_reflect], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing initial values, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")

    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    for i in range(self.max_sc):
        thinking_sc, answer_sc = await cot_sc_agents[i]([taskInfo, thinking_reflect, thinking_1_2], cot_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, refining initial computations, thinking: {thinking_sc.content}; answer: {answer_sc.content}")
        possible_answers_1_2.append(answer_sc)
        possible_thinkings_1_2.append(thinking_sc)

    thinking_1_2_final, answer_1_2_final = await final_decision_agent_0_1([taskInfo] + possible_thinkings_1_2, "Sub-task 2: Synthesize and finalize initial computations.", is_sub_task=True)

    reflect_inst_1_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_2 = "Sub-task 2 (Reflexion): Your problem is to compute and tabulate winning/losing status for n=1 to 10." + reflect_inst_1_2

    thinking_reflect_1_2, answer_reflect_1_2 = await cot_agent([taskInfo] + possible_thinkings_1_2 + [thinking_1_2_final], cot_reflect_instruction_1_2, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, refining computations, thinking: {thinking_reflect_1_2.content}; answer: {answer_reflect_1_2.content}")

    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking_reflect_1_2], critic_inst_1_1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        thinking_reflect_1_2, answer_reflect_1_2 = await cot_agent([taskInfo, thinking_reflect_1_2, feedback], cot_reflect_instruction_1_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining computations, thinking: {thinking_reflect_1_2.content}; answer: {answer_reflect_1_2.content}")

    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_reflect_1_2.content}; answer - {answer_reflect_1_2.content}")
    logs.append({"subtask_id": "stage_1.subtask_2", "instruction": cot_instruction_1_2, "response": {"thinking": thinking_reflect_1_2, "answer": answer_reflect_1_2}, "agent_collaboration": "CoT | SC_CoT | Reflexion"})
    print("Step 1.2: ", sub_tasks[-1])

    # Subtask 3: Analyze pattern or periodicity to derive closed-form (depends on stage_1.subtask_2) (CoT | SC_CoT | Reflexion)
    cot_instruction_1_3 = "Sub-task 3: Analyze the pattern or periodicity found in the winning/losing positions to derive a closed-form characterization or modular arithmetic condition for losing positions."
    thinking_1_3, answer_1_3 = await cot_agent([taskInfo, thinking_reflect_1_2], cot_instruction_1_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing pattern, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")

    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    for i in range(self.max_sc):
        thinking_sc, answer_sc = await cot_sc_agents[i]([taskInfo, thinking_reflect_1_2, thinking_1_3], cot_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, refining pattern analysis, thinking: {thinking_sc.content}; answer: {answer_sc.content}")
        possible_answers_1_3.append(answer_sc)
        possible_thinkings_1_3.append(thinking_sc)

    thinking_1_3_final, answer_1_3_final = await final_decision_agent_0_1([taskInfo] + possible_thinkings_1_3, "Sub-task 3: Synthesize and finalize pattern characterization.", is_sub_task=True)

    reflect_inst_1_3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_3 = "Sub-task 3 (Reflexion): Your problem is to analyze and characterize the pattern of losing positions." + reflect_inst_1_3

    thinking_reflect_1_3, answer_reflect_1_3 = await cot_agent([taskInfo] + possible_thinkings_1_3 + [thinking_1_3_final], cot_reflect_instruction_1_3, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, refining pattern characterization, thinking: {thinking_reflect_1_3.content}; answer: {answer_reflect_1_3.content}")

    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking_reflect_1_3], critic_inst_1_1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        thinking_reflect_1_3, answer_reflect_1_3 = await cot_agent([taskInfo, thinking_reflect_1_3, feedback], cot_reflect_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining pattern characterization, thinking: {thinking_reflect_1_3.content}; answer: {answer_reflect_1_3.content}")

    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_reflect_1_3.content}; answer - {answer_reflect_1_3.content}")
    logs.append({"subtask_id": "stage_1.subtask_3", "instruction": cot_instruction_1_3, "response": {"thinking": thinking_reflect_1_3, "answer": answer_reflect_1_3}, "agent_collaboration": "CoT | SC_CoT | Reflexion"})
    print("Step 1.3: ", sub_tasks[-1])

    # Stage 2

    # Subtask 1: Count number of losing positions for Alice (depends on stage_1.subtask_3) (CoT | Reflexion)
    cot_instruction_2_1 = "Sub-task 1: Using the characterization from stage_1.subtask_3, count the number of losing positions for Alice (i.e., winning positions for Bob) for all n from 1 to 2024."
    thinking_2_1, answer_2_1 = await cot_agent([taskInfo, thinking_reflect_1_3], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, counting losing positions, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")

    reflect_inst_2_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_1 = "Sub-task 1 (Reflexion): Your problem is to count losing positions for Alice from 1 to 2024." + reflect_inst_2_1

    thinking_reflect_2_1, answer_reflect_2_1 = await cot_agent([taskInfo, thinking_2_1], cot_reflect_instruction_2_1, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, refining count, thinking: {thinking_reflect_2_1.content}; answer: {answer_reflect_2_1.content}")

    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking_reflect_2_1], critic_inst_1_1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        thinking_reflect_2_1, answer_reflect_2_1 = await cot_agent([taskInfo, thinking_reflect_2_1, feedback], cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining count, thinking: {thinking_reflect_2_1.content}; answer: {answer_reflect_2_1.content}")

    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_reflect_2_1.content}; answer - {answer_reflect_2_1.content}")
    logs.append({"subtask_id": "stage_2.subtask_1", "instruction": cot_instruction_2_1, "response": {"thinking": thinking_reflect_2_1, "answer": answer_reflect_2_1}, "agent_collaboration": "CoT | Reflexion"})
    print("Step 2.1: ", sub_tasks[-1])

    # Subtask 2: Verify correctness of counting method and final count (depends on stage_2.subtask_1) (CoT | Reflexion | Debate)
    cot_instruction_2_2 = "Sub-task 2: Verify the correctness of the counting method and the final count by cross-checking with sample computations or alternative reasoning."
    thinking_2_2, answer_2_2 = await cot_agent([taskInfo, thinking_reflect_2_1], cot_instruction_2_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, verifying count, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")

    reflect_inst_2_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_2 = "Sub-task 2 (Reflexion): Your problem is to verify the counting method and final count." + reflect_inst_2_2

    thinking_reflect_2_2, answer_reflect_2_2 = await cot_agent([taskInfo, thinking_2_2], cot_reflect_instruction_2_2, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, refining verification, thinking: {thinking_reflect_2_2.content}; answer: {answer_reflect_2_2.content}")

    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking_reflect_2_2], critic_inst_1_1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        thinking_reflect_2_2, answer_reflect_2_2 = await cot_agent([taskInfo, thinking_reflect_2_2, feedback], cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining verification, thinking: {thinking_reflect_2_2.content}; answer: {answer_reflect_2_2.content}")

    debate_instr_2_2 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_2_2 = "Sub-task 2: Verify correctness of counting method and final count." + debate_instr_2_2

    all_thinking_2_2 = [[] for _ in range(self.max_round)]
    all_answer_2_2 = [[] for _ in range(self.max_round)]

    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking_d, answer_d = await agent([taskInfo, thinking_reflect_2_2], debate_instruction_2_2, r, is_sub_task=True)
            else:
                thinking_d, answer_d = await agent([taskInfo, thinking_reflect_2_2] + all_thinking_2_2[r-1], debate_instruction_2_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_d.content}; answer: {answer_d.content}")
            all_thinking_2_2[r].append(thinking_d)
            all_answer_2_2[r].append(answer_d)

    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_final_2_2, answer_final_2_2 = await final_decision_agent_2_2([taskInfo] + all_thinking_2_2[-1], "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking_final_2_2.content}; answer: {answer_final_2_2.content}")

    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_final_2_2.content}; answer - {answer_final_2_2.content}")
    logs.append({"subtask_id": "stage_2.subtask_2", "instruction": cot_instruction_2_2, "response": {"thinking": thinking_final_2_2, "answer": answer_final_2_2}, "agent_collaboration": "CoT | Reflexion | Debate"})
    print("Step 2.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_final_2_2, answer_final_2_2, sub_tasks, agents)
    return final_answer, logs
